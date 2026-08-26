# NEXUS-Engine — образ для обучения и инференса (CPU-профиль).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXUS_REGISTRY=/data/registry

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY nexus ./nexus
RUN pip install --upgrade pip && pip install -e "."

COPY examples ./examples
COPY tests ./tests
COPY docs ./docs
COPY nexus.sh Makefile ./

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fs http://localhost:8000/health || exit 1

# по умолчанию — API; для обучения: docker run ... nexus quickstart --scale small
ENTRYPOINT ["python", "-m", "nexus.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--registry", "/data/registry"]
