# Prediction service image. It runs the exported ONNX model through onnxruntime, so it
# needs neither PyTorch nor CUDA and stays small. The model is baked in; export it first
# with `python scripts/export_onnx.py`, which writes models/yolo11n_kitti.onnx.
FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless drops the GUI libraries but still links libglib at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so the layer caches across source changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY app ./app
COPY models/yolo11n_kitti.onnx ./models/yolo11n_kitti.onnx

# /app puts the app package on the path; /app/src puts drive_perception on it.
ENV PYTHONPATH=/app:/app/src
ENV DRIVE_WEIGHTS=/app/models/yolo11n_kitti.onnx

EXPOSE 8000
CMD ["uvicorn", "app.service:app", "--host", "0.0.0.0", "--port", "8000"]
