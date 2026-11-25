# Ming Advisor Acupoint Server

Copy the existing `hand_landmarker.task` and `face_landmarker.task` into this directory before deploying. They are not included in this package.

Render: New Web Service, Docker runtime, Dockerfile at `./Dockerfile`, health check `/health`, one instance. Set `ACUPOINT_API_KEY` to a random demo key in the service environment; never commit the key.

`GET /health` → status ok. `POST /detect` → raw JPEG, `Content-Type: image/jpeg`, `X-API-Key` header. Response schema is unchanged from the existing Unity integration.

The single detector handles one request at a time; HTTP 503 means busy. No frames or coordinates are intentionally persisted or logged by the app. This is a controlled demo, not a production authentication solution. Obtain user consent before sending camera frames to the cloud.
