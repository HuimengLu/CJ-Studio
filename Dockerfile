# CJ Studio — FastAPI backend image.
# Deployable to Hugging Face Spaces (Docker SDK, default port 7860), Render,
# or Cloud Run (both set $PORT). The Next.js frontend deploys separately on
# Vercel and proxies /api/* here via its BACKEND_URL.
FROM python:3.11-slim

# System libraries:
#   opencv-python-headless → libgl1, libglib2.0-0
#   Pillow                 → libjpeg62-turbo, libpng16-16, libwebp7
#   text-overlay fonts     → fonts-dejavu-core, fonts-liberation
#   healthcheck / TLS      → curl, ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
        fonts-dejavu-core \
        fonts-liberation \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Hugging Face Spaces runs the container as uid 1000; give that user a writable
# home so the withoutbg model cache and any temp files land somewhere valid.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CJ_UPSCALE=0
WORKDIR /home/user/app

# App code + the runtime assets the pipeline reads (see .dockerignore for what
# stays out — the frontend and the legacy Streamlit app are not needed here).
COPY --chown=user backend/ backend/
COPY --chown=user static/social/ static/social/
COPY --chown=user fonts/ fonts/
COPY --chown=user bg_artwork.png .

USER user

# Bake the background-removal weights into the image so the first request is
# fast and startup needs no network (best effort — the app re-downloads at
# runtime if this step is skipped).
RUN python -c "from withoutbg import WithoutBG; WithoutBG.opensource()" || true

EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --timeout-keep-alive 75"]
