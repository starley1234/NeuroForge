FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_QPA_PLATFORM=offscreen \
    NEUROSCAD_RENDER_TIMEOUT=30

RUN apt-get update && apt-get install -y --no-install-recommends \
      openscad ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY neuroscad ./neuroscad
COPY schema ./schema
COPY requirements-prod.lock ./
RUN pip install --no-cache-dir -r requirements-prod.lock && pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 10001 neuroscad
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=4s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)" || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "neuroscad.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips=*"]
