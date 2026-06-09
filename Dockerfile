# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 – builder
#   Install Python packages (with build tools available) and pre-download the
#   rembg ONNX model so the runtime image never hits the network at startup.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries required by:
#   opencv-python-headless  → libgomp1, libglib2.0-0
#   Pillow                  → libjpeg62-turbo, libpng16-16, libwebp7
#   onnxruntime             → libgomp1 (OpenMP)
#   model download          → ca-certificates (HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the rembg ONNX model into /rembg_models so the runtime stage
# never makes outbound network calls after the image is built.
# Override the model via build arg:  --build-arg REMBG_MODEL=isnet-general-use
ARG REMBG_MODEL=u2net
ENV U2NET_HOME=/rembg_models
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python - <<'PYEOF'
import os, sys
sys.path.insert(0, "/install/lib/python3.11/site-packages")
model = os.environ.get("REMBG_MODEL", "u2net")
print(f"Pre-downloading rembg model: {model}")
from rembg import new_session
new_session(model)
print("Model cached to", os.environ["U2NET_HOME"])
PYEOF

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 – runtime
#   Lean image: only what's needed to run the app, no build tools.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

WORKDIR /app

# Same runtime libraries (no -dev packages needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded rembg model — no network call on first request
COPY --from=builder /rembg_models /rembg_models
ENV U2NET_HOME=/rembg_models

# Copy application (excludes paths in .dockerignore)
COPY . .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Liveness probe: Streamlit exposes a health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run as non-root
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /rembg_models
USER appuser

ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableStaticServing=true"]
