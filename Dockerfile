FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/app/artifacts

WORKDIR /app

COPY requirements.lock.txt .
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.lock.txt

ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

COPY train_singapore.py qualifying_grid.example.json singapore_predictions.json singapore_metadata.json singapore_model.pkl singapore_inference.json ./
COPY tests ./tests

CMD ["python", "train_singapore.py"]
