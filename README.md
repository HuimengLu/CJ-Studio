# CJ Studio

Internal tools for Construction Junction: product-photo enhancement (listing
formatter) and a social-media graphic generator.

## Architecture

- **`backend/`** — FastAPI API (Python). Owns all image processing:
  background removal (withoutbg), optional Real-ESRGAN upscale, canvas
  composition, text overlay, and the Figma-derived social templates.
  - `pipeline.py` — listing pipeline (framework-free)
  - `social_engine.py` — social template renderer (framework-free)
  - `main.py` — HTTP endpoints
- **`frontend/`** — Next.js app (TypeScript, App Router). All UI:
  - `/` — Listing workflow: multi-photo upload → batch processing →
    before/after compare, per-photo ratio/text-overlay edits, filmstrip,
    export selection (PNG/ZIP)
  - `/social` — Social Media Generator: ratio+upload → template gallery →
    live text editing → download
- **`app.py`** — legacy Streamlit app (same features, pre-migration). Kept
  temporarily; will be removed once the new stack is fully adopted.

## Run (development)

```bash
# 1. backend (port 8000)
pip install -r requirements.txt
uvicorn backend.main:app --port 8000 --timeout-keep-alive 75

# 2. frontend (port 3000; /api/* proxies to :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:3000.

## Deployment

The two halves deploy independently:

- **Backend** — any Python host with ≥4 GB RAM (Hugging Face Spaces, Render,
  Cloud Run…). First request downloads the withoutbg weights from Hugging
  Face. `models/RealESRGAN_x4plus.pth` (64 MB, git-ignored) enables AI
  upscaling; without it the pipeline falls back to LANCZOS.
- **Frontend** — Vercel / any Node host, or `next build` output on a static
  host. Set `BACKEND_URL` so `/api/*` rewrites point at the deployed backend.

## Assets

- `static/social/` — decoration assets exported from the Figma file
  (regenerate alphas with `python3 scripts/prep_social_assets.py`)
- `fonts/Gabarito.ttf` — brand font used by the social templates
- `bg_artwork.png` — text-overlay background artwork
