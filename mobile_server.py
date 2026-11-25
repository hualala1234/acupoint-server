"""Cloud demo API for Ming Advisor. Images are processed in memory."""
import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

ACUPOINTS = {

    "shenmen": {
        "chinese_name": "神門穴",
        "category": "A",
        "category_name": "壓力型",
        "symptom": "壓力超載、情緒煩躁",
        "location": "手腕橫紋處，靠近小指側的凹陷處。",
        "type": "hand",
        "points": [0, 17],
        "ratio": 0.82
    },

    "neiguan": {
        "chinese_name": "內關穴",
        "category": "B",
        "category_name": "焦慮型",
        "symptom": "焦慮不安、心跳加速",
        "location": "手腕內側中心，距離橫紋約三指寬處。",
        "type": "hand",
        "points": [0, 9],
        "ratio": 0.28
    },

    "hegu": {
        "chinese_name": "合谷穴",
        "category": "C",
        "category_name": "疲勞型",
        "symptom": "體力倦怠、精神不濟",
        "location": "手背虎口處，第一、二掌骨間肌肉最高點。",
        "type": "hand",
        "points": [2, 5],
        "ratio": 0.65
    },

    "baihui": {
        "chinese_name": "百會穴",
        "category": "D",
        "category_name": "穩定型",
        "symptom": "心境平穩、需強化專注",
        "location": "頭頂正中央，兩耳尖連線的交點。",
        "type": "face",
        "points": [10]
    },

    "yintang": {
        "chinese_name": "印堂穴",
        "category": "E",
        "category_name": "失眠型",
        "symptom": "慢性疲勞、睡眠障礙",
        "location": "兩眉頭連線的中點（即眉心）。",
        "type": "face",
        "points": [168]
    }
}

class Detector:
    def __init__(self):
        root = Path(__file__).resolve().parent
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(root / "hand_landmarker.task")),
                running_mode=vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5))
        try:
            self.face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(root / "face_landmarker.task")),
                    running_mode=vision.RunningMode.IMAGE, num_faces=1,
                    min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5))
        except Exception:
            self.hand.close()
            raise

    def close(self):
        self.hand.close()
        self.face.close()

    def detect(self, img):
        # Unity sends the already rotated/mirrored preview. Do not flip again.
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = time.time_ns() // 1_000_000
        hand = self.hand.detect(image)
        face = self.face.detect(image)
        points = {}

        def put(key, pos):
            x, y = pos
            info = ACUPOINTS[key]
            points[key] = dict(id=key, name=info["chinese_name"],
                               english_name=key, category=info["category"],
                               symptom=info["symptom"], location=info["location"],
                               x=x, y=y, normalized_x=x/w, normalized_y=y/h)

        def weighted(a, b, r):
            return (int(a[0]*(1-r)+b[0]*r), int(a[1]*(1-r)+b[1]*r))

        # Preserve the original one-marker-per-acupoint behavior (last hand wins).
        for landmarks in hand.hand_landmarks:
            p = [(int(v.x*w), int(v.y*h)) for v in landmarks]
            put("shenmen", weighted(p[17], p[0], ACUPOINTS["shenmen"]["ratio"]))
            put("neiguan", weighted(p[0], p[9], ACUPOINTS["neiguan"]["ratio"]))
            put("hegu", weighted(p[2], p[5], ACUPOINTS["hegu"]["ratio"]))
        for landmarks in face.face_landmarks:
            p = [(int(v.x*w), int(v.y*h)) for v in landmarks]
            face_height = abs(p[152][1] - p[10][1])
            put("baihui", (p[10][0], max(0, min(h-1, int(p[10][1]-face_height*0.18)))))
            put("yintang", p[168])
        return dict(frame_width=w, frame_height=h, timestamp=ts,
                    acupoints=list(points.values()))


MAX_BYTES = 2_000_000
MAX_PIXELS = 2_000_000

@asynccontextmanager
async def lifespan(app: FastAPI):
    token = os.environ.get("ACUPOINT_API_KEY", "")
    if not token:
        raise RuntimeError("ACUPOINT_API_KEY must be configured")
    cv2.setNumThreads(1)
    app.state.token = token.encode("utf-8")
    app.state.lock = asyncio.Lock()
    app.state.detector = Detector()
    try:
        yield
    finally:
        app.state.detector.close()

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

@app.get("/health")
async def health():
    return {"status": "ok"}


def decode_and_detect(detector, raw):
    try:
        # Validate dimensions before OpenCV allocates the decoded image.
        with Image.open(BytesIO(raw)) as image:
            if image.format != "JPEG" or image.width * image.height > MAX_PIXELS:
                raise ValueError("Invalid JPEG or resolution")
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[0] * img.shape[1] > MAX_PIXELS:
            raise ValueError("Invalid JPEG or resolution")
    except Exception:
        raise HTTPException(400, "Invalid JPEG or resolution") from None
    try:
        return detector.detect(img)
    except Exception:
        # Do not log uploaded images or landmark coordinates.
        raise HTTPException(500, "Detection failed") from None

@app.post("/detect")
async def detect(request: Request):
    key = request.headers.get("X-API-Key", "").encode("utf-8")
    if not hmac.compare_digest(key, request.app.state.token):
        raise HTTPException(401, "Unauthorized")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
        raise HTTPException(415, "Expected image/jpeg")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            n = int(length)
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length") from None
        if n <= 0 or n > MAX_BYTES:
            raise HTTPException(413, "Invalid image size")
    lock = request.app.state.lock
    if lock.locked():
        raise HTTPException(503, "Detector busy; retry later")
    async with lock:
        raw = bytearray()
        try:
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > MAX_BYTES:
                        raise HTTPException(413, "Image too large")
                    raw.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, "Upload timed out") from None
        if not raw:
            raise HTTPException(400, "Empty image")
        return await run_in_threadpool(decode_and_detect, request.app.state.detector, raw)
