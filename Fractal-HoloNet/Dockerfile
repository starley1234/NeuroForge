FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TEXT_MODEL_DIR=/app/checkpoints/fractal_holonet_base \
    SIGNAL_MODEL_DIR=/app/checkpoints/fractal_holonet_multimodal \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and checkpoints
COPY . .

# NOTE: чекпоинты уже скопированы из репозитория (COPY . .).
# Случайная модель здесь НЕ создаётся: serve.py сам инициализирует модель,
# только если каталог чекпоинта действительно отсутствует.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "fractal_holonet.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
