"""CJ Studio API — FastAPI backend for the Next.js frontend.

Listing workflow
    POST   /api/photos                     multipart file → process, store, return id
    GET    /api/photos/{id}/image          composed result (ratio/text/caption/w params)
    GET    /api/photos/{id}/thumb          small JPEG of the enhanced result
    DELETE /api/photos/{id}
    POST   /api/export                     selected photos → single PNG or ZIP

Social workflow
    POST   /api/social/upload              raw image → id
    GET    /api/social/templates           templates compatible with ?ratio=
    GET    /api/social/render              PNG preview/final for img+template+texts

State is in-memory (single-process deployment, mirrors the old Streamlit
session): photos evicted after PHOTO_TTL seconds of inactivity.
"""
import io
import logging
import os
import time
import uuid
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import pipeline, social_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cj_api")

app = FastAPI(title="CJ Studio API")

# Dev: Next.js runs on :3000 and proxies /api/* here, so CORS is belt-and-
# braces for direct calls (e.g. curl, other ports).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTO_TTL = 60 * 60 * 6      # 6h — internal tool, generous
_photos: dict = {}           # id → record
_social_imgs: dict = {}      # id → {bytes, touched}
_testing2: dict = {}         # id → {panels, pngs, touched}


def _evict_stale() -> None:
    now = time.time()
    for store in (_photos, _social_imgs, _testing2):
        for k in [k for k, v in store.items() if now - v["touched"] > PHOTO_TTL]:
            store.pop(k, None)


def _get_photo(photo_id: str) -> dict:
    rec = _photos.get(photo_id)
    if rec is None:
        raise HTTPException(404, "photo not found")
    rec["touched"] = time.time()
    return rec


# ── Listing: process / compose / export ────────────────────────────────────────

@app.post("/api/photos")
async def create_photo(file: UploadFile = File(...)):
    _evict_stale()
    raw = await file.read()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    except Exception:
        raise HTTPException(400, f"{file.filename}: could not open image")

    result, warn, mask, rgba, orig_layer = pipeline.make_listing(img)
    if warn:
        return JSONResponse({"id": None, "name": file.filename, "warn": warn},
                            status_code=422)

    photo_id = uuid.uuid4().hex[:12]
    _photos[photo_id] = {
        "name": file.filename,
        "original": img,
        "result": result,
        "subject_mask": mask,
        "subject_rgba": rgba,
        "orig_layer": orig_layer,
        "compose_cache": {},
        "touched": time.time(),
    }
    return {"id": photo_id, "name": file.filename, "warn": None}


def _composed_png(rec: dict, ratio: str, text_mode: bool, caption: str,
                  kind: str, width: int | None, orig_bg: bool = False) -> bytes:
    key = (kind, ratio, text_mode, caption, width, orig_bg)
    cache = rec["compose_cache"]
    if key in cache:
        return cache[key]

    if kind == "before":
        img = pipeline.fit_to_ratio(rec["original"], ratio)
    else:
        img = pipeline.compose(rec["result"], rec["subject_rgba"],
                               ratio, text_mode, caption,
                               rec.get("orig_layer") if orig_bg else None)
    if width and width < img.width:
        img = img.resize((width, round(img.height * width / img.width)),
                         Image.LANCZOS)
    png = pipeline.to_png_bytes(img)
    if len(cache) > 16:
        cache.clear()
    cache[key] = png
    return png


@app.get("/api/photos/{photo_id}/image")
def photo_image(photo_id: str, ratio: str = "1:1", text: int = 0,
                caption: str = "", kind: str = "after", w: int | None = None,
                origbg: int = 0):
    rec = _get_photo(photo_id)
    png = _composed_png(rec, ratio, bool(text), caption, kind, w, bool(origbg))
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/photos/{photo_id}/thumb")
def photo_thumb(photo_id: str, s: int = 160):
    rec = _get_photo(photo_id)
    thumb = rec["result"].copy()
    thumb.thumbnail((s, s))
    return Response(pipeline.to_jpeg_bytes(thumb, quality=82),
                    media_type="image/jpeg")


@app.delete("/api/photos/{photo_id}")
def delete_photo(photo_id: str):
    _photos.pop(photo_id, None)
    return {"ok": True}


class ExportItem(BaseModel):
    id: str
    ratio: str = "1:1"
    text_mode: bool = False
    caption: str = ""
    orig_bg: bool = False


class ExportRequest(BaseModel):
    items: list[ExportItem]


@app.post("/api/export")
def export_photos(req: ExportRequest):
    if not req.items:
        raise HTTPException(400, "nothing selected")

    def _png(item: ExportItem) -> tuple[str, bytes]:
        rec = _get_photo(item.id)
        png = _composed_png(rec, item.ratio, item.text_mode, item.caption,
                            "after", None, item.orig_bg)
        stem = (rec["name"].rsplit(".", 1)[0] or "photo")
        return f"cj_{stem}_{item.ratio.replace(':', 'x')}.png", png

    if len(req.items) == 1:
        name, png = _png(req.items[0])
        return Response(png, media_type="image/png", headers={
            "Content-Disposition": f'attachment; filename="{name}"'})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        used = set()
        for item in req.items:
            name, png = _png(item)
            while name in used:                      # dedupe repeated stems
                name = name.replace(".png", "_1.png")
            used.add(name)
            zf.writestr(name, png)
    return Response(buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="cj_photos.zip"'})


# ── Testing 2: shadow-method comparison ─────────────────────────────────────────

@app.post("/api/testing2")
async def create_testing2(file: UploadFile = File(...)):
    _evict_stale()
    raw = await file.read()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    except Exception:
        raise HTTPException(400, f"{file.filename}: could not open image")

    panels, warn = pipeline.make_listing_variants(img)
    if warn:
        return JSONResponse({"id": None, "warn": warn}, status_code=422)

    tid = uuid.uuid4().hex[:12]
    _testing2[tid] = {
        "meta": [{"key": k, "label": lbl, "desc": d} for k, lbl, d, _ in panels],
        "pngs": {k: pipeline.to_png_bytes(im) for k, _, _, im in panels},
        "touched": time.time(),
    }
    return {"id": tid, "name": file.filename,
            "panels": _testing2[tid]["meta"], "warn": None}


@app.get("/api/testing2/{tid}/{key}")
def testing2_image(tid: str, key: str, w: int | None = None):
    rec = _testing2.get(tid)
    if rec is None or key not in rec["pngs"]:
        raise HTTPException(404, "not found")
    rec["touched"] = time.time()
    png = rec["pngs"][key]
    if w:
        img = Image.open(io.BytesIO(png))
        if w < img.width:
            img = img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)
            png = pipeline.to_png_bytes(img)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ── Social ─────────────────────────────────────────────────────────────────────

@app.post("/api/social/upload")
async def social_upload(file: UploadFile = File(...)):
    _evict_stale()
    raw = await file.read()
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(400, "could not open image")
    img_id = uuid.uuid4().hex[:12]
    _social_imgs[img_id] = {"bytes": raw, "touched": time.time()}
    return {"id": img_id}


@app.get("/api/social/templates")
def social_templates(ratio: str = "4:5"):
    if ratio not in social_engine.RATIO_BY_ID:
        raise HTTPException(400, f"unknown ratio {ratio}")
    return {"ratio": social_engine.RATIO_BY_ID[ratio] | {},
            "templates": social_engine.templates_for(ratio)}


@app.get("/api/social/recommend")
def social_recommend(ratio: str = "4:5", title: str = "", subtitle: str = "",
                     theme: str = "", count: int = 3, seed: int | None = None):
    """Recommend 2-3 templates that fit the uploaded content.

    Presence of `title`/`subtitle` text drives compatibility; `theme` pins a
    preferred colour (otherwise randomised); `seed` makes a shortlist
    reproducible (omit it to re-roll a fresh set on each call).
    """
    if ratio not in social_engine.RATIO_BY_ID:
        raise HTTPException(400, f"unknown ratio {ratio}")
    recs = social_engine.recommend(
        has_image=True,
        has_title=bool(title.strip()),
        has_subtitle=bool(subtitle.strip()),
        ratio_id=ratio,
        preferred_theme=theme.strip() or None,
        count=max(1, min(count, 3)),
        seed=seed,
    )
    return {"ratio": ratio, "themes": social_engine.THEMES,
            "recommendations": recs}


@app.get("/api/social/render")
def social_render(img: str, template: str, ratio: str = "4:5",
                  title: str = "", subtitle: str = "", theme: str = "",
                  w: int = 540):
    rec = _social_imgs.get(img)
    if rec is None:
        raise HTTPException(404, "image not found")
    rec["touched"] = time.time()
    if template not in social_engine.TEMPLATE_BY_ID:
        raise HTTPException(400, f"unknown template {template}")
    if ratio not in social_engine.RATIO_BY_ID:
        raise HTTPException(400, f"unknown ratio {ratio}")
    w = max(96, min(w, 2048))
    png = social_engine.render_template(
        template, ratio, rec["bytes"],
        {"title": title, "subtitle": subtitle}, w, theme=theme.strip() or None)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ── Testing (fixed-image Style-1 preview) ────────────────────────────────────────

_TESTING_IMG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "testing", "fixed.jpg")
_testing_img_cache: dict = {}


def _testing_img() -> bytes:
    """The fixed sample photo used by the Testing preview (loaded once)."""
    if "bytes" not in _testing_img_cache:
        with open(_TESTING_IMG_PATH, "rb") as f:
            _testing_img_cache["bytes"] = f.read()
    return _testing_img_cache["bytes"]


@app.get("/api/testing/render")
def testing_render(style: str = "style1", theme: str = "green",
                   ratio: str = "1:1", title: str = "", subtitle: str = "",
                   w: int = 720):
    """Render the fixed image in one Testing style + theme + ratio with text.

    Empty title falls back to 'Construction Junction'; empty subtitle is omitted
    (both handled inside the renderer).
    """
    if style not in social_engine.TESTING_STYLES:
        raise HTTPException(400, f"unknown style {style}")
    if theme not in social_engine.TESTING_THEMES:
        raise HTTPException(400, f"unknown theme {theme}")
    if ratio not in social_engine.STYLE1_H:
        raise HTTPException(400, f"unknown ratio {ratio}")
    w = max(96, min(w, 2048))
    png = social_engine.render_testing(style, theme, ratio, _testing_img(),
                                       title, subtitle, w)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"ok": True, "photos": len(_photos), "social": len(_social_imgs)}
