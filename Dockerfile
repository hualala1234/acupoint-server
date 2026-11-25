FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libglib2.0-0 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mobile_server.py hand_landmarker.task face_landmarker.task ./
CMD ["sh", "-c", "exec uvicorn mobile_server:app --host 0.0.0.0 --port ${PORT:-8008} --workers 1 --no-access-log"]
