import base64
import functools
import io
import json
import logging
import math
import os
import textwrap
import time
import zipfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cj_listing")

# ── background removal ─────────────────────────────────────────────────────────
# withoutbg (open source) — Depth-Anything V2 + ISNet segmentation + focus
# matting/refiner. Produces a soft 0-255 alpha matte with cleaner edges than
# plain u2net. Models download once from Hugging Face and are cached locally.
# https://github.com/withoutbg/withoutbg

# ── page config ────────────────────────────────────────────────────────────────
# ?view=social shows the embedded IAAC tool; default is the Listing formatter.
_VIEW = st.query_params.get("view", "listing")

st.set_page_config(
    page_title="CJ Studio",
    layout="wide" if _VIEW == "social" else "centered",
    initial_sidebar_state="collapsed",
)

# ── Lumina-inspired palette (editorial monochrome) ─────────────────────────────
INK        = "#1b1b1b"   # near-black, primary
INK_SOFT   = "#5d5e66"   # secondary text
MUTED      = "#9a9a9a"   # tertiary / overline labels
LINE       = "#cfc4c5"   # hairline / outline-variant
CARD       = "#ffffff"   # surface-container-lowest
CHIP       = "#e1e3e4"   # surface-container-highest
CHIP_HOVER = "#d4d6d7"
BG         = "#f8f9fa"    # background / surface
DARK       = INK         # retained: referenced by the page heading below

st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@300;400;500;600;700&family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
  [data-testid="stAppViewContainer"] {{ background: {BG}; }}
  html, body, [data-testid="stAppViewContainer"], .stApp,
  [data-testid="stMarkdownContainer"], [data-testid="stFileUploader"] {{
    font-family: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  .block-container {{ max-width: 880px; padding-top: 2rem; padding-bottom: 3rem; }}
  header[data-testid="stHeader"] {{ background: transparent; box-shadow: none; }}
  /* No stray loading UI: hide Streamlit's top-right RUNNING / status widget.
     Processing state is shown only by the in-place progress bar. */
  [data-testid="stStatusWidget"] {{ display: none !important; }}
  /* Don't fade the whole app to grey while a rerun computes (stale dimming). */
  [data-stale="true"] {{ opacity: 1 !important; }}
  .stApp [data-stale="true"] {{ opacity: 1 !important; }}

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {{
    border: 1.5px dashed {LINE};
    border-radius: 0;
    padding: 2.5rem 1rem;
    background: {CARD};
    text-align: center;
  }}
  [data-testid="stFileUploader"] section {{ background: transparent; border: none; }}

  /* ── Buttons ── */
  .stDownloadButton > button,
  .stButton > button {{
    border: 1px solid transparent !important;
    border-radius: 0 !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    white-space: nowrap !important;
  }}
  /* Secondary / default buttons */
  .stButton > button {{
    background: {CHIP} !important;
    color: {INK} !important;
  }}
  .stButton > button:hover {{
    background: {CHIP_HOVER} !important;
    color: {INK} !important;
  }}
  /* Primary buttons (active ratio chip) + download button */
  .stDownloadButton > button,
  .stButton > button[kind="primary"] {{
    background: {INK} !important;
    color: #ffffff !important;
  }}
  .stDownloadButton > button:hover,
  .stButton > button[kind="primary"]:hover {{
    background: #000000 !important;
    color: #ffffff !important;
  }}
  /* Ratio chip row: tighter padding, smaller font */
  .ratio-row .stButton > button {{
    padding: 0.4rem 0 !important;
    font-size: 0.82rem !important;
  }}
  /* Text inputs: ink focus ring instead of theme red */
  [data-testid="stTextInput"] [data-baseweb="input"] {{ border-radius: 0; }}
  [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{
    border-color: {INK} !important;
  }}
  /* Hide the built-in "Press Enter to apply" hint (Apply button instead) */
  [data-testid="InputInstructions"] {{ display: none; }}
  /* ── After image lightbox ── */
  .cj-after {{ position:relative; display:block; width:100%; }}
  .cj-after > img {{ width:100%; border-radius:8px; display:block; cursor:zoom-in; }}
  .cj-expand-btn {{
    position:absolute; top:0.4rem; right:0.4rem;
    opacity:0; transition:opacity 0.15s;
    background:white; border:none; border-radius:6px;
    width:2rem; height:2rem; padding:0;
    display:flex; align-items:center; justify-content:center;
    cursor:pointer; box-shadow:0 1px 4px rgba(0,0,0,0.18);
    pointer-events:auto;
  }}
  .cj-after:hover .cj-expand-btn {{ opacity:1; }}
  #cj-lb {{
    display:none; position:fixed; top:0; left:0;
    width:100vw; height:100vh;
    background:rgba(0,0,0,0.88); z-index:99999;
    justify-content:center; align-items:center;
  }}
  #cj-lb > img {{
    max-width:90vw; max-height:90vh;
    border-radius:8px; object-fit:contain;
  }}
  .cj-lb-dl {{
    position:fixed; top:1.25rem; right:1.25rem;
    background:{INK}; color:#fff;
    border-radius:6px; padding:0.5rem 1.4rem;
    font-weight:600; font-size:0.82rem; letter-spacing:0.03em;
    text-decoration:none; white-space:nowrap;
  }}
  .cj-lb-dl:hover {{ background:#000; color:#fff; }}
  .col-label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 8px;
  }}
  .warn-box {{
    background: #fffbe6;
    border: 1px solid #ffe58f;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #7d6608;
    font-size: 0.88rem;
    margin: 0.8rem 0;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# ── withoutbg model (loaded once, cached for the lifetime of the server) ───────

@st.cache_resource(show_spinner=False)
def _withoutbg_model():
    from withoutbg import WithoutBG

    log.info("Loading withoutbg open-source model...")
    t0 = time.perf_counter()
    try:
        model = WithoutBG.opensource()
    except Exception as exc:
        log.error("Failed to initialise withoutbg model: %s", exc, exc_info=True)
        raise RuntimeError(
            "Could not load the background-removal model. "
            "Check your internet connection — the first run downloads the "
            f"model weights from Hugging Face. Detail: {exc}"
        ) from exc

    log.info("withoutbg model ready in %.1f s", time.perf_counter() - t0)
    return model


# ── Real-ESRGAN upscaler (optional, local only) ────────────────────────────────
# Subjects whose longest side is below this get an AI upscale instead of a blurry
# LANCZOS enlarge before they're blown up to fill the canvas.
_UPSCALE_BELOW = int(os.environ.get("CJ_UPSCALE_BELOW", "900"))
# Weights + torch are NOT in requirements.txt: if either is absent (e.g. the
# deployed build) the loader returns None and the pipeline falls back to LANCZOS.
_ESRGAN_ENABLED = os.environ.get("CJ_UPSCALE", "1") != "0"


@st.cache_resource(show_spinner=False)
def _esrgan_model():
    """Load Real-ESRGAN x4 (via spandrel) onto MPS/CUDA/CPU.

    Returns (net, device, scale) or None when unavailable — torch/spandrel not
    installed or the weight file is missing — so background removal still works
    without the (heavy) upscaling dependency.
    """
    if not _ESRGAN_ENABLED:
        return None
    try:
        import torch
        from spandrel import ModelLoader
    except Exception as exc:
        log.info("Real-ESRGAN unavailable (%s) — skipping AI upscale.", exc)
        return None

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()
    weight = os.path.join(base_dir, "models", "RealESRGAN_x4plus.pth")
    if not os.path.exists(weight):
        log.info("Real-ESRGAN weight not found at %s — skipping AI upscale.", weight)
        return None

    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = ModelLoader().load_from_file(weight)
        net = model.model.eval().to(device)
        log.info("Real-ESRGAN x%s ready on %s", model.scale, device)
        return net, device, int(model.scale)
    except Exception as exc:
        log.warning("Failed to load Real-ESRGAN: %s", exc, exc_info=True)
        return None


def _esrgan_upscale(rgb: np.ndarray):
    """Upscale an HxWx3 uint8 array with Real-ESRGAN.

    Returns (rgb_up, factor). Falls back to (rgb, 1) when the model is
    unavailable or inference fails, so the caller can proceed with LANCZOS.
    """
    loaded = _esrgan_model()
    if loaded is None:
        return rgb, 1
    net, device, factor = loaded
    try:
        import torch

        t0 = time.perf_counter()
        x = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        )
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        if device == "mps":
            torch.mps.synchronize()
        out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        log.info(
            "Real-ESRGAN upscaled %dx%d → %dx%d in %.2f s",
            rgb.shape[1], rgb.shape[0], out.shape[1], out.shape[0],
            time.perf_counter() - t0,
        )
        return out, factor
    except Exception as exc:
        log.warning("Real-ESRGAN inference failed (%s) — using LANCZOS.", exc)
        return rgb, 1


# ── image processing ───────────────────────────────────────────────────────────

def _extract_subject(pil_img: Image.Image):
    """
    Remove background with withoutbg and return (rgb, alpha) as uint8 arrays.
    alpha is a smooth 0-255 matte — no hard thresholding done here.
    """
    log.info(
        "Removing background from %dx%d image using withoutbg...",
        pil_img.width, pil_img.height,
    )
    t0 = time.perf_counter()

    try:
        rgba = _withoutbg_model().remove_background(pil_img.convert("RGB"))
    except RuntimeError:
        raise
    except MemoryError as exc:
        log.error("OOM during withoutbg inference: %s", exc, exc_info=True)
        raise RuntimeError(
            "Not enough memory to process this image. "
            "Try a smaller image (< 4000 x 4000 px)."
        ) from exc
    except Exception as exc:
        log.error("withoutbg remove_background() failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"Background removal failed ({type(exc).__name__}). "
            "Try a different image or restart the app."
        ) from exc

    log.info("Background removed in %.2f s", time.perf_counter() - t0)
    arr = np.array(rgba.convert("RGBA"))
    return arr[:, :, :3], arr[:, :, 3]


def _tighten_alpha(alpha: np.ndarray) -> np.ndarray:
    """
    Post-process the model's alpha matte:
    1. Remove isolated noise specks (small open/close)
    2. Fill topologically enclosed holes (e.g. ring centre, frame window)
    3. Hard-zero pixels below confidence threshold
       – the matte can assign alpha 0-80 to background gaps between limbs;
         zeroing them forces those gaps to be fully transparent on the
         canvas (white) rather than showing a faint background colour.
    Soft product edges (alpha ≥ 80) are preserved for natural blending.
    """
    # ── 1. Binary mask + tiny speck cleanup ──────────────────────────────────
    _, binary = cv2.threshold(alpha, 15, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)

    # ── 2. Fill topologically enclosed holes ─────────────────────────────────
    inv = cv2.bitwise_not(binary)
    _, labels = cv2.connectedComponents(inv, connectivity=8)
    border_labels = set(np.concatenate([
        labels[0, :], labels[-1, :],
        labels[1:-1, 0], labels[1:-1, -1],
    ]).tolist())
    filled = binary.copy()
    for lbl in np.unique(labels):
        if lbl != 0 and lbl not in border_labels:
            filled[labels == lbl] = 255     # enclosed hole → opaque

    # ── 3. Re-apply alpha; fix holes; hard-zero low-confidence pixels ────────
    # Pixels inside enclosed holes had alpha=0 → set to 255 (product colour)
    # Pixels outside the mask → 0
    # Background-gap pixels (alpha 1-79) → 0 (transparent = canvas white)
    out = np.where(
        filled > 0,
        np.where(alpha > 0, alpha, np.uint8(255)),
        np.uint8(0),
    ).astype(np.uint8)
    out[out < 80] = 0   # kill semi-transparent background residue in gaps
    return out


def _color_guided_cleanup(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Second-pass cleanup: zero moderate-alpha pixels (80–210) whose colour
    closely matches the detected background colour.

    This catches concave "gap" areas (e.g. between arm and body) that the model
    assigns surprisingly high alpha because they're surrounded by product
    pixels, yet their actual colour is the background.

    Skipped automatically when the background is near-white (mean ≥ 230) to
    avoid incorrectly removing white parts of the product.
    """
    h, w = rgb.shape[:2]
    b = max(10, min(30, h // 20, w // 20))

    # ── Estimate background colour from image border strips ──────────────────
    strips = [rgb[:b, :], rgb[-b:, :], rgb[:, :b], rgb[:, -b:]]
    border_px = np.concatenate([s.reshape(-1, 3) for s in strips]).astype(np.float32)
    if len(border_px) == 0:
        return alpha

    bg_color = np.median(border_px, axis=0)   # (3,) float32

    # Skip if background is near-white — colour distance too unreliable
    if float(np.mean(bg_color)) > 230:
        return alpha

    # ── Per-pixel L2 colour distance from background ─────────────────────────
    diff = rgb.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))   # (h, w)

    # Zero alpha where: uncertain zone (80–210) AND colour ≈ background
    out = alpha.copy()
    uncertain = (alpha >= 80) & (alpha <= 210)
    out[uncertain & (dist < 45)] = 0
    return out


def generate_product_shadow(r_mask, bx_s, by_s, bw_s, bh_s,
                             ox, oy, target_cx, CS):
    """Simple soft drop shadow: blurred silhouette shifted slightly down-right."""
    H, W = r_mask.shape[:2]
    empty = np.zeros((CS, CS), dtype=np.float32)

    if int(np.count_nonzero(r_mask > 128)) < 100 or bw_s < 4 or bh_s < 4:
        return empty, empty

    sh_x = max(1, int(bw_s * 0.03))
    sh_y = max(1, int(bh_s * 0.02))

    shadow = empty.copy()
    s_oy, s_ox = oy + sh_y, ox + sh_x
    cy0 = max(0, s_oy);  cy1 = min(CS, s_oy + H)
    cx0 = max(0, s_ox);  cx1 = min(CS, s_ox + W)
    if cy1 > cy0 and cx1 > cx0:
        iy0 = cy0 - s_oy;  iy1 = iy0 + (cy1 - cy0)
        ix0 = cx0 - s_ox;  ix1 = ix0 + (cx1 - cx0)
        shadow[cy0:cy1, cx0:cx1] = r_mask[iy0:iy1, ix0:ix1].astype(np.float32) / 255.0

    shadow = cv2.GaussianBlur(shadow, (0, 0), 24.0)
    shadow = np.clip(shadow * 0.16, 0.0, 1.0)
    return shadow, empty


def make_listing(pil_img: Image.Image):
    """
    Full pipeline: withoutbg subject extraction → crop → 1600×1600 canvas + shadow.
    Returns (result_pil, warning_str | None).
    """
    log.info(
        "make_listing: start input=%dx%d",
        pil_img.width, pil_img.height,
    )
    t_total = time.perf_counter()

    try:
        rgb, alpha = _extract_subject(pil_img)
    except RuntimeError as exc:
        log.warning("make_listing: subject extraction failed - %s", exc)
        return None, str(exc)

    log.info("make_listing: running alpha post-processing...")
    alpha = _tighten_alpha(alpha)
    alpha = _color_guided_cleanup(rgb, alpha)   # zero gap pixels matching bg colour

    h, w = rgb.shape[:2]
    total = h * w

    fg_px = int(np.count_nonzero(alpha > 15))
    if fg_px < total * 0.01:
        log.warning("make_listing: foreground too small (fg_px=%d / total=%d)", fg_px, total)
        return None, "Could not detect the item. Try a photo with a cleaner background."

    pts = cv2.findNonZero((alpha > 15).astype(np.uint8) * 255)
    if pts is None:
        log.warning("make_listing: findNonZero returned None")
        return None, "Could not detect the item. Try a photo with a cleaner background."
    bx, by, bw, bh = cv2.boundingRect(pts)

    # Crop with a small margin around the tight bounding box
    pad = max(10, int(max(bw, bh) * 0.04))
    x1, y1 = max(0, bx - pad), max(0, by - pad)
    x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
    crop_rgb   = rgb[y1:y2, x1:x2]
    crop_alpha = alpha[y1:y2, x1:x2]

    # ── Canvas ──────────────────────────────────────────────────────────────
    CS = 1600
    canvas = np.full((CS, CS, 3), [253, 253, 242], dtype=np.uint8)  # #FDFDF2

    # Scale so the SUBJECT's longest side fills 75 % of the canvas.
    # Using bw/bh (not crop dims) ensures the item is never scaled down
    # just because the crop contains transparent margins.
    scale  = (CS * 0.75) / max(bw, bh)

    # Low-res subject → AI-upscale the crop first so it stays crisp when enlarged,
    # instead of a soft LANCZOS blow-up. Dividing `scale` by the upscale factor
    # keeps the final on-canvas size (and the centring maths below) unchanged.
    up = 1
    if max(bw, bh) < _UPSCALE_BELOW:
        crop_rgb, up = _esrgan_upscale(crop_rgb)
        if up > 1:
            crop_alpha = cv2.resize(
                crop_alpha, (crop_rgb.shape[1], crop_rgb.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

    ch_c, cw_c = crop_rgb.shape[:2]
    nw = max(1, int(cw_c * scale / up))
    nh = max(1, int(ch_c * scale / up))
    r_rgb  = cv2.resize(crop_rgb,   (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    r_mask = cv2.resize(crop_alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # Subject centre in the scaled crop
    scx = int((bx - x1 + bw / 2) * scale)
    scy = int((by - y1 + bh / 2) * scale)

    # Target: horizontal centre, 52 % from top (natural product-photo feel)
    target_cx = CS // 2
    target_cy = int(CS * 0.52)

    # Top-left of the scaled crop on the canvas (can be negative → clipped)
    ox = target_cx - scx
    oy = target_cy - scy

    # Helper: returns matching canvas slice and crop slice
    def _slices(offset: int, length: int, limit: int):
        c0 = max(0, offset)
        c1 = min(limit, offset + length)
        return slice(c0, c1), slice(c0 - offset, c1 - offset)

    can_y, img_y = _slices(oy, nh, CS)
    can_x, img_x = _slices(ox, nw, CS)

    # ── Shadow generation ─────────────────────────────────────────────────────
    # Scaled bounding-box coordinates within the scaled crop (r_mask space)
    bx_s = int((bx - x1) * scale)
    by_s = int((by - y1) * scale)
    bw_s = int(bw * scale)
    bh_s = int(bh * scale)

    cast_shadow, contact_shadow = generate_product_shadow(
        r_mask, bx_s, by_s, bw_s, bh_s, ox, oy, target_cx, CS,
    )

    # Compositing order (per spec):
    #   1. background  (canvas, already filled)
    #   2. cast shadow
    #   3. contact shadow
    #   4. foreground object
    total_shadow = np.clip(cast_shadow + contact_shadow, 0.0, 1.0)
    canvas_f = canvas.astype(np.float32)
    for c in range(3):
        canvas_f[:, :, c] = np.clip(canvas_f[:, :, c] * (1.0 - total_shadow), 0, 255)
    canvas = canvas_f.astype(np.uint8)

    # ── Composite item ───────────────────────────────────────────────────────
    alpha_f = r_mask[img_y, img_x].astype(np.float32) / 255.0
    roi = canvas[can_y, can_x].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = r_rgb[img_y, img_x, c] * alpha_f + roi[:, :, c] * (1.0 - alpha_f)
    canvas[can_y, can_x] = roi.astype(np.uint8)

    # ── Store full-canvas subject mask + RGBA for text overlay ───────────────
    full_mask = np.zeros((CS, CS), dtype=np.uint8)
    full_mask[can_y, can_x] = r_mask[img_y, img_x]

    full_rgba = np.zeros((CS, CS, 4), dtype=np.uint8)
    full_rgba[can_y, can_x, :3] = r_rgb[img_y, img_x]
    full_rgba[can_y, can_x,  3] = r_mask[img_y, img_x]

    st.session_state["subject_mask"] = full_mask
    st.session_state["subject_rgba"] = full_rgba

    # Subtle brightness + contrast lift
    out = Image.fromarray(canvas)
    out = ImageEnhance.Brightness(out).enhance(1.02)
    out = ImageEnhance.Contrast(out).enhance(1.05)
    log.info(
        "make_listing: done in %.2f s output=%dx%d",
        time.perf_counter() - t_total, out.width, out.height,
    )
    return out, None


def _to_png_bytes(img: Image.Image, optimize: bool = False) -> bytes:
    # optimize=True roughly doubles encode time; the on-screen preview / download
    # don't need max compression, so default to the fast path.
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=optimize)
    return buf.getvalue()


# ── Text overlay styles ──────────────────────────────────────────────────────────

TEXT_STYLES = {
    "Bold & Loud": {
        "desc": "Big chunky letters, crayon-sketch feel",
        "emoji": "📢",
    },
    "Clean & Minimal": {
        "desc": "Simple sans-serif, quiet and modern",
        "emoji": "✦",
    },
    "Handwritten": {
        "desc": "Casual brushstroke energy",
        "emoji": "✍️",
    },
    "Retro Stamp": {
        "desc": "Vintage badge with distressed edges",
        "emoji": "🔖",
    },
}


def _load_font(size: int, style: str) -> ImageFont.ImageFont:
    """Try to load a suitable system font for each style, fall back to default."""
    candidates = {
        "Bold & Loud": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/Helvetica Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ],
        "Clean & Minimal": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ],
        "Handwritten": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
            "/Library/Fonts/Brush Script MT Italic.ttf",
            "/Library/Fonts/Apple Chancery.ttf",
            "/System/Library/Fonts/Supplemental/Brush Script MT Italic.ttf",
        ],
        "Retro Stamp": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/Helvetica Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ],
    }
    for path in candidates.get(style, []):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _render_bold_loud(canvas: Image.Image, text: str, CS: int, font_size: int) -> Image.Image:
    """
    Bold & Loud: oversized all-caps text split across two lines, rotated -8°,
    drawn in vibrant green with a strong outline to fill the background.
    """
    layer = Image.new("RGBA", (CS, CS), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)

    words  = text.upper().split()
    chunks = _split_words(words, 2)
    n      = len(chunks)

    font_size = max(140, min(font_size, int(CS * 0.65)))
    font      = _load_font(font_size, "Bold & Loud")

    for _ in range(24):
        widths = [draw.textlength(c, font=font) for c in chunks]
        if max(widths) < CS * 0.95:
            break
        font_size = max(140, int(font_size * 0.94))
        font = _load_font(font_size, "Bold & Loud")

    bbox   = font.getbbox("Ay")
    lh     = bbox[3] - bbox[1] + int(font_size * 0.08)
    y      = int(CS * 0.12)

    for i, line in enumerate(chunks):
        w = draw.textlength(line, font=font)
        x = int(CS * 0.06) if i == 0 else int(CS * 0.05)

        for dx, dy in [(-14, 0), (14, 0), (0, -14), (0, 14), (-8, -8), (8, 8)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(210, 235, 95, 240))
        draw.text((x, y), line, font=font, fill=(18, 75, 20, 255))

        for offset in range(0, int(w), 12):
            draw.line(
                [(x + offset, y + lh // 2), (x + offset + 14, y + lh // 2 - 10)],
                fill=(190, 220, 80, 120), width=8,
            )
        y += lh

    layer = layer.rotate(-8, resample=Image.BICUBIC, expand=False)
    out   = canvas.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def _render_background_text(canvas: Image.Image, CS: int) -> Image.Image:
    """
    Draw a two-line Construction Junction background text layer behind the subject.
    """
    lines = ["CONSTRUCTION", "JUNCTION"]
    layer = Image.new("RGBA", (CS, CS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font_size = int(CS * 0.18)
    font = _load_font(font_size, "Bold & Loud")
    for _ in range(20):
        widths = [draw.textlength(line, font=font) for line in lines]
        if max(widths) < CS * 0.96:
            break
        font_size = max(140, int(font_size * 0.94))
        font = _load_font(font_size, "Bold & Loud")

    bbox = font.getbbox("Ay")
    line_height = bbox[3] - bbox[1] + int(font_size * 0.08)
    total_height = line_height * len(lines)
    y = int(CS * 0.08)

    for line in lines:
        width = draw.textlength(line, font=font)
        x = (CS - width) // 2
        draw.text((x, y), line, font=font, fill=(210, 235, 95, 180))
        y += line_height

    out = canvas.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def _render_clean_minimal(canvas: Image.Image, text: str, CS: int) -> Image.Image:
    """
    Clean & Minimal: centred text in light gray, large but understated.
    Split into short lines. No rotation.
    """
    layer = Image.new("RGBA", (CS, CS), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)

    words  = text.upper().split()
    chunks = _split_words(words, 3)
    n      = len(chunks)

    font_size = max(110, CS // (n + 2))
    font      = _load_font(font_size, "Clean & Minimal")

    for _ in range(20):
        widths = [draw.textlength(c, font=font) for c in chunks]
        if max(widths) < CS * 0.88:
            break
        font_size = int(font_size * 0.88)
        font = _load_font(font_size, "Clean & Minimal")

    bbox = font.getbbox("Ay")
    lh   = bbox[3] - bbox[1] + int(font_size * 0.18)
    total_h = lh * n
    y    = (CS - total_h) // 2

    for line in chunks:
        w = draw.textlength(line, font=font)
        x = (CS - w) // 2
        draw.text((x, y), line, font=font, fill=(160, 158, 148, 180))
        y += lh

    out = canvas.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def _render_handwritten(canvas: Image.Image, text: str, CS: int) -> Image.Image:
    """
    Handwritten: italic, slightly varied line heights, warm green ink,
    bold and textured with a gentle 3° tilt.
    """
    layer = Image.new("RGBA", (CS, CS), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)

    words  = text.upper().split()
    chunks = _split_words(words, 3)
    n      = len(chunks)

    font_size = max(140, CS // (n + 1))
    font      = _load_font(font_size, "Handwritten")

    for _ in range(20):
        widths = [draw.textlength(c, font=font) for c in chunks]
        if max(widths) < CS * 0.85:
            break
        font_size = int(font_size * 0.88)
        font = _load_font(font_size, "Handwritten")

    bbox  = font.getbbox("Ay")
    lh    = bbox[3] - bbox[1] + int(font_size * 0.15)
    total_h = lh * n
    y     = (CS - total_h) // 2

    rng = np.random.default_rng(42)
    for i, line in enumerate(chunks):
        w    = draw.textlength(line, font=font)
        x    = (CS - w) // 2 + int(rng.integers(-28, 28))
        jitter = int(rng.integers(-12, 12))
        # rough textured outline for marker effect
        for dx, dy in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
            draw.text((x + dx, y + jitter + dy), line, font=font, fill=(45, 90, 35, 120))
        draw.text((x, y + jitter), line, font=font, fill=(25, 70, 25, 230))
        draw.line([(x - 10, y + lh - 8 + jitter),
                   (x + int(w) + 10, y + lh - 10 + jitter)],
                  fill=(40, 80, 40, 80), width=6)
        y += lh

    layer = layer.rotate(3, resample=Image.BICUBIC, expand=False)
    out   = canvas.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def _render_retro_stamp(canvas: Image.Image, text: str, CS: int) -> Image.Image:
    """
    Retro Stamp: text inside a distressed circular badge overlay,
    dark red ink, uppercase, centred.
    """
    layer = Image.new("RGBA", (CS, CS), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)

    # ── Badge circle ──────────────────────────────────────────────────────────
    cx, cy = CS // 2, CS // 2
    r_outer = int(CS * 0.42)
    r_inner = int(CS * 0.36)
    ink = (160, 30, 20, 180)

    draw.ellipse([(cx - r_outer, cy - r_outer), (cx + r_outer, cy + r_outer)],
                 outline=ink, width=10)
    draw.ellipse([(cx - r_inner, cy - r_inner), (cx + r_inner, cy + r_inner)],
                 outline=ink, width=4)

    # ── Text ─────────────────────────────────────────────────────────────────
    words  = text.upper().split()
    chunks = _split_words(words, 3)
    n      = len(chunks)

    font_size = max(90, int(r_inner * 1.2 // (n + 0.5)))
    font      = _load_font(font_size, "Retro Stamp")

    for _ in range(20):
        widths = [draw.textlength(c, font=font) for c in chunks]
        if max(widths) < r_inner * 1.7:
            break
        font_size = int(font_size * 0.88)
        font = _load_font(font_size, "Retro Stamp")

    bbox    = font.getbbox("Ay")
    lh      = bbox[3] - bbox[1] + int(font_size * 0.12)
    total_h = lh * n
    y       = cy - total_h // 2

    for line in chunks:
        w = draw.textlength(line, font=font)
        x = cx - w // 2
        draw.text((x + 3, y + 3), line, font=font, fill=(160, 30, 20, 60))  # shadow
        draw.text((x, y), line, font=font, fill=ink)
        y += lh

    # Distress: random noise spots to simulate worn ink
    rng = np.random.default_rng(7)
    arr = np.array(layer)
    for _ in range(300):
        px = int(rng.integers(0, CS))
        py = int(rng.integers(0, CS))
        if arr[py, px, 3] > 50:
            arr[py, px, 3] = max(0, int(arr[py, px, 3]) - int(rng.integers(80, 180)))
    layer = Image.fromarray(arr)

    layer = layer.rotate(-5, resample=Image.BICUBIC, expand=False)
    out   = canvas.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def _split_words(words: list, max_lines: int) -> list:
    """Split a word list into at most max_lines roughly-equal chunks."""
    if not words:
        return [""]
    n = min(len(words), max_lines)
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


def apply_text_overlay(base: Image.Image, target_ratio: str = "1:1", description_text: str = "") -> Image.Image:
    """
    Composite text (background) + shadow + scaled-down centered subject.
    The subject is pulled from subject_rgba (clean, no shadow) and repositioned.
    target_ratio: used to position artwork so it doesn't get clipped during cropping.
    description_text: if provided, used to center the entire composition vertically.
    """
    CS = base.width   # 1600
    bg_color = (253, 253, 242)

    # Start with solid colored canvas (keeps background visible under transparent artwork)
    from PIL import Image as _PILImage
    canvas_img = _PILImage.new("RGBA", (CS, CS), tuple(bg_color) + (255,))

    # First pass: Calculate dimensions and vertical offset for centering composition
    # Get artwork dimensions
    artwork_w, artwork_h = 0, 0
    if st.session_state.get("bg_image") is not None:
        bg = st.session_state.get("bg_image")
        bw, bh = bg.size
        
        ar = _RATIO_AR.get(target_ratio, 1.0)
        if ar >= 1.0:
            safe_width = CS
            safe_height = int(CS / ar)
        else:
            safe_width = int(CS * ar)
            safe_height = CS
        
        safe_margin = 0.02
        max_w = int(safe_width * (1.0 - 2 * safe_margin))
        max_h = int(safe_height * (1.0 - 2 * safe_margin))
        scale = min(max_w / float(bw), max_h / float(bh))
        artwork_w = max(1, int(bw * scale))
        artwork_h = max(1, int(bh * scale))

    # Get subject dimensions
    subject_rgba = st.session_state.get("subject_rgba")
    subject_w, subject_h = 0, 0
    if subject_rgba is not None:
        scale_factor = 0.94
        subject_h, subject_w = subject_rgba.shape[:2]
        subject_w = max(1, int(subject_w * scale_factor))
        subject_h = max(1, int(subject_h * scale_factor))

    # Estimate description height
    desc_height = 0
    if description_text.strip():
        font_size = max(32, int(CS * 0.04 * 0.4))
        line_height = int(font_size * 1.2)
        max_width = int(CS * 0.9)
        word_count = len(description_text.split())
        avg_chars_per_line = max(3, max_width // (font_size * 0.6))
        estimated_words_per_line = max(1, int(avg_chars_per_line / 5))
        estimated_lines = max(1, (word_count + estimated_words_per_line - 1) // estimated_words_per_line)
        desc_height = int(line_height * estimated_lines + CS * 0.03)

    # Calculate vertical offset to center entire composition
    artwork_gap = int(CS * 0.03)
    subject_gap = int(CS * 0.03)
    total_comp_height = artwork_h + artwork_gap + subject_h + subject_gap + desc_height
    available_space = CS - total_comp_height
    vertical_offset = max(0, available_space // 2)

    # Now composite artwork with offset
    if st.session_state.get("bg_image") is not None:
        bg = st.session_state.get("bg_image")
        bw, bh = bg.size
        
        ar = _RATIO_AR.get(target_ratio, 1.0)
        if ar >= 1.0:
            safe_width = CS
            safe_height = int(CS / ar)
        else:
            safe_width = int(CS * ar)
            safe_height = CS
        
        safe_margin = 0.02
        max_w = int(safe_width * (1.0 - 2 * safe_margin))
        max_h = int(safe_height * (1.0 - 2 * safe_margin))
        scale = min(max_w / float(bw), max_h / float(bh))
        new_w = max(1, int(bw * scale))
        new_h = max(1, int(bh * scale))
        
        try:
            resized = bg.resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            resized = bg.resize((new_w, new_h))
        rgba = resized.convert("RGBA")
        
        # Position artwork at top with centering offset
        x = (CS - new_w) // 2
        y = vertical_offset + max(int(CS * 0.02), int(safe_height * safe_margin))
        canvas_img.paste(rgba, (x, y), rgba)
        
        # Store artwork bounds for description positioning
        st.session_state["artwork_bottom_y"] = y + new_h
    else:
        # Render generated text artwork and composite it over the colored canvas
        base_rgb = np.full((CS, CS, 3), bg_color, dtype=np.uint8)
        text_art = _render_background_text(_PILImage.fromarray(base_rgb), CS)
        text_rgba = text_art.convert("RGBA")
        canvas_img.paste(text_rgba, (0, 0), text_rgba)

    canvas = np.array(canvas_img.convert("RGB"))

    # Get the clean subject RGBA from session state
    subject_rgba = st.session_state.get("subject_rgba")
    if subject_rgba is None:
        # Fallback: just return text+subject as before
        result = Image.fromarray(canvas)
        result = ImageEnhance.Brightness(result).enhance(1.02)
        result = ImageEnhance.Contrast(result).enhance(1.05)
        return result

    # Scale down the subject to 94% of original size
    scale_factor = 0.94
    subject_h, subject_w = subject_rgba.shape[:2]
    new_w = max(1, int(subject_w * scale_factor))
    new_h = max(1, int(subject_h * scale_factor))
    
    scaled_rgb = cv2.resize(subject_rgba[:, :, :3], (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    scaled_alpha = cv2.resize(subject_rgba[:, :, 3], (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Position subject with vertical centering applied
    center_x = CS // 2
    ox = center_x - new_w // 2
    
    # Calculate subject Y position based on artwork presence
    # Use actual artwork bounds from session if available, otherwise centered with offset
    artwork_bottom_stored = st.session_state.get("artwork_bottom_y", 0)
    if artwork_bottom_stored > 0:
        # Artwork was placed; position subject below it
        subject_gap = int(CS * 0.03)
        oy = artwork_bottom_stored + subject_gap
    else:
        # No artwork; use vertical centering with small offset
        center_y = CS // 2
        oy = center_y - new_h // 2
        oy += vertical_offset - int(CS * 0.04)  # Apply vertical offset adjustment
    
    # Ensure subject stays within canvas bounds
    oy = max(0, min(oy, CS - new_h))

    # Generate shadow for the scaled subject
    bx_s = 0
    by_s = 0
    bw_s = new_w
    bh_s = new_h
    cast_shadow, _ = generate_product_shadow(scaled_alpha, bx_s, by_s, bw_s, bh_s, ox, oy, center_x, CS)

    # Composite shadow onto canvas (text layer)
    total_shadow = np.clip(cast_shadow, 0.0, 1.0)
    canvas_f = canvas.astype(np.float32)
    for c in range(3):
        canvas_f[:, :, c] = np.clip(canvas_f[:, :, c] * (1.0 - total_shadow), 0, 255)
    canvas = canvas_f.astype(np.uint8)

    # Composite scaled subject on top
    def _slices(offset: int, length: int, limit: int):
        c0 = max(0, offset)
        c1 = min(limit, offset + length)
        return slice(c0, c1), slice(c0 - offset, c1 - offset)

    can_y, img_y = _slices(oy, new_h, CS)
    can_x, img_x = _slices(ox, new_w, CS)

    alpha_f = scaled_alpha[img_y, img_x].astype(np.float32) / 255.0
    roi = canvas[can_y, can_x].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = scaled_rgb[img_y, img_x, c] * alpha_f + roi[:, :, c] * (1.0 - alpha_f)
    canvas[can_y, can_x] = roi.astype(np.uint8)

    # Store the bottom edge of the subject for description positioning
    subject_bottom = can_y.stop if hasattr(can_y, 'stop') else (oy + new_h)
    st.session_state["subject_bottom_y"] = subject_bottom

    result = Image.fromarray(canvas)
    result = ImageEnhance.Brightness(result).enhance(1.02)
    result = ImageEnhance.Contrast(result).enhance(1.05)
    return result


def _add_description_text(img: Image.Image, text: str) -> Image.Image:
    """
    Add small description text at the bottom of the image.
    """
    if not text.strip():
        return img
    
    W, H = img.size
    draw = ImageDraw.Draw(img)
    
    # Load a smaller font for description
    font_size = max(32, int(W * 0.04))
    font = _load_font(font_size, "Clean & Minimal")
    
    # Wrap text if needed
    max_width = int(W * 0.9)
    lines = []
    current_line = ""
    for word in text.split():
        test_line = current_line + (" " if current_line else "") + word
        if draw.textlength(test_line, font=font) > max_width:
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    
    # Draw text positioned based on product bottom edge (detected from overlay)
    line_height = int(font_size * 1.2)
    total_height = line_height * len(lines)
    
    # Use actual stored bounds from apply_text_overlay
    subject_bottom = st.session_state.get("subject_bottom_y", H)
    
    # Scale from 1600x1600 to the current cropped image size
    full_size = 1600
    scale_factor = H / float(full_size) if H < full_size else 1.0
    adjusted_subject_bottom = int(subject_bottom * scale_factor)
    
    # Position description just below the product with a small gap
    gap = int(W * 0.03)
    y = min(adjusted_subject_bottom + gap, H - total_height - int(W * 0.02))
    y = max(y, int(W * 0.01))  # ensure it's not too close to top
    
    for line in lines:
        width = draw.textlength(line, font=font)
        x = (W - width) // 2
        draw.text((x, y), line, font=font, fill=(100, 100, 100, 255))
        y += line_height
    
    return img


def _composite_by_color_diff(original: Image.Image,
                              text_canvas: Image.Image,
                              bg_color: tuple) -> Image.Image:
    """Fallback: detect subject by colour distance from bg, re-composite."""
    orig_arr = np.array(original).astype(np.float32)
    bg       = np.array(bg_color, dtype=np.float32)
    diff     = np.sqrt(((orig_arr - bg) ** 2).sum(axis=2))
    mask     = (diff > 18).astype(np.float32)
    # Soft-erode so we don't grab bg pixels near subject
    mask_u8  = (mask * 255).astype(np.uint8)
    k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8  = cv2.erode(mask_u8, k, iterations=1)
    mask     = mask_u8.astype(np.float32) / 255.0

    text_arr = np.array(text_canvas).astype(np.float32)
    for c in range(3):
        text_arr[:, :, c] = (orig_arr[:, :, c] * mask
                             + text_arr[:, :, c] * (1.0 - mask))
    return Image.fromarray(text_arr.astype(np.uint8))


_RATIO_AR = {"1:1": 1.0, "4:3": 4/3, "4:5": 4/5}


def _detect_rgba_bounds(rgba_array: np.ndarray) -> tuple:
    """
    Detect the bounding box of non-transparent pixels in an RGBA array.
    Returns (top, bottom, left, right) or None if all transparent.
    """
    if rgba_array.shape[2] < 4:
        # No alpha channel
        return (0, rgba_array.shape[0], 0, rgba_array.shape[1])
    
    alpha = rgba_array[:, :, 3]
    # Find rows and cols with alpha > 0
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None  # All transparent
    
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    return (row_indices[0], row_indices[-1] + 1, col_indices[0], col_indices[-1] + 1)


def _corner_color(img: Image.Image):
    """Average the four corner pixels — the padding colour, so bars are seamless."""
    w, h = img.size
    px = img.load()
    pts = [(1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2)]
    cols = [px[x, y] for x, y in pts]
    n = len(cols[0]) if isinstance(cols[0], tuple) else 1
    if n == 1:
        return int(round(sum(cols) / len(cols)))
    return tuple(int(round(sum(c[i] for c in cols) / len(cols))) for i in range(n))


def _fit_to_ratio(img: Image.Image, ratio: str) -> Image.Image:
    """Pad img with its own background colour to the target aspect ratio.

    Never crops: for a wider target we add left/right bars, for a taller one we
    add top/bottom bars, so the subject and any text overlay stay fully visible.
    """
    W, H = img.size
    ar = _RATIO_AR.get(ratio, 1.0)
    cur = W / float(H)
    if abs(cur - ar) < 1e-3:
        return img
    if ar > cur:                      # need wider → pad left/right
        new_w, new_h = int(round(H * ar)), H
    else:                             # need taller → pad top/bottom
        new_w, new_h = W, int(round(W / ar))
    canvas = Image.new(img.mode, (new_w, new_h), _corner_color(img))
    canvas.paste(img, ((new_w - W) // 2, (new_h - H) // 2))
    return canvas


# ── session state ──────────────────────────────────────────────────────────────
# Each uploaded photo is an independent record in st.session_state.photos; its
# per-image edit state (ratio / text overlay / caption) lives in the record, so
# editing one photo never touches another. `active` is the index shown on the
# result screen. The singular keys below (ratio, text_mode, subject_rgba …) are
# the *active mirror*: they reflect the active photo so the existing rendering
# pipeline (make_listing / apply_text_overlay) works unchanged.
for _k in ("original", "result", "warn", "subject_mask", "subject_rgba"):
    if _k not in st.session_state:
        st.session_state[_k] = None
if "photos" not in st.session_state:
    st.session_state.photos = []          # list[dict] — see _new_photo()
if "active" not in st.session_state:
    st.session_state.active = 0
if "show_export" not in st.session_state:
    st.session_state.show_export = False
if "export_sel" not in st.session_state:
    st.session_state.export_sel = set()   # indices selected in the export modal
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None   # photo index pending delete confirm
if "ratio" not in st.session_state:
    st.session_state.ratio = "1:1"
if "text_mode" not in st.session_state:
    st.session_state.text_mode = False
if "bg_image" not in st.session_state:
    st.session_state.bg_image = None
    # Auto-load project background artwork if present at repo root named 'bg_artwork.png'
    try:
        base_dir = os.path.dirname(__file__)
    except Exception:
        base_dir = os.getcwd()
    default_bg = os.path.join(base_dir, "bg_artwork.png")
    if os.path.exists(default_bg):
        try:
            st.session_state.bg_image = ImageOps.exif_transpose(Image.open(default_bg)).convert("RGBA")
        except Exception:
            st.session_state.bg_image = Image.open(default_bg).convert("RGBA")
if "description_text" not in st.session_state:
    st.session_state.description_text = ""
# The caption actually rendered on the image — only updated when "Apply" is clicked.
if "applied_description" not in st.session_state:
    st.session_state.applied_description = ""


# ── Per-photo records ──────────────────────────────────────────────────────────
def _new_photo(name, original, result, subject_rgba, subject_mask):
    """Build one photo record. Edit state defaults to the current active mirror
    so a newly added photo inherits the last-used ratio, matching user intent."""
    return {
        "name": name,
        "original": original,
        "result": result,
        "subject_rgba": subject_rgba,
        "subject_mask": subject_mask,
        "ratio": st.session_state.get("ratio", "1:1"),
        "text_mode": False,
        "applied_description": "",
        "description_text": "",
    }


def _active_photo():
    photos = st.session_state.photos
    if not photos:
        return None
    i = min(st.session_state.active, len(photos) - 1)
    return photos[i]


def _sync_active_mirror(p):
    """Point the singular pipeline keys + widget-bound keys at photo `p`."""
    st.session_state.ratio = p["ratio"]
    st.session_state.text_mode = p["text_mode"]
    st.session_state.cj_txt = p["text_mode"]
    st.session_state.description_text = p["description_text"]
    st.session_state.applied_description = p["applied_description"]
    st.session_state.subject_rgba = p["subject_rgba"]
    st.session_state.subject_mask = p["subject_mask"]


# ── Widget callbacks — mutate state *before* the rerun so the canvas at the top
#    of the result screen recomputes exactly once (no wasted extra rerun). ───────
def _activate(i):
    """Switch the active photo, saving the current caption draft first."""
    photos = st.session_state.photos
    cur = _active_photo()
    if cur is not None:
        cur["description_text"] = st.session_state.get("description_text", "")
    st.session_state.active = max(0, min(i, len(photos) - 1))
    _sync_active_mirror(photos[st.session_state.active])

def _pick_ratio(r):
    st.session_state.ratio = r
    p = _active_photo()
    if p is not None:
        p["ratio"] = r

def _sync_text_mode():
    st.session_state.text_mode = st.session_state.cj_txt
    p = _active_photo()
    if p is not None:
        p["text_mode"] = st.session_state.cj_txt

def _apply_caption():
    st.session_state.applied_description = st.session_state.description_text
    p = _active_photo()
    if p is not None:
        p["applied_description"] = st.session_state.description_text
        p["description_text"] = st.session_state.description_text

def _reset_photos():
    st.session_state.photos = []
    st.session_state.active = 0
    st.session_state.warn = None
    st.session_state.result = None

def _ask_delete(i):
    st.session_state.confirm_delete = i

def _delete_photo(i):
    """Remove photo i and repoint the active index / mirror."""
    photos = st.session_state.photos
    if not (0 <= i < len(photos)):
        return
    photos.pop(i)
    if not photos:
        _reset_photos()
        return
    a = st.session_state.active
    if i < a:
        a -= 1
    a = max(0, min(a, len(photos) - 1))
    st.session_state.active = a
    st.session_state._before_cache = {}   # cache keys are index-based
    _sync_active_mirror(photos[a])

def _open_export():
    st.session_state.show_export = True
    st.session_state.export_sel = set(range(len(st.session_state.photos)))

def _toggle_export(i):
    sel = st.session_state.export_sel
    sel.discard(i) if i in sel else sel.add(i)


def _compose_photo(p) -> Image.Image:
    """Render a photo record to its final image at its own ratio + text state.

    apply_text_overlay / _add_description_text read the subject arrays from the
    singular session keys, so point those at `p` first. Safe because Streamlit
    runs one script pass at a time per session.
    """
    st.session_state["subject_rgba"] = p["subject_rgba"]
    st.session_state["subject_mask"] = p["subject_mask"]
    base = p["result"]
    caption = p["applied_description"] if p["text_mode"] else ""
    if p["text_mode"]:
        square = apply_text_overlay(base, "1:1", caption)
    else:
        square = base
    if caption.strip():
        square = _add_description_text(square, caption.strip())
    return _fit_to_ratio(square, p["ratio"])


def _process_one(f):
    """Run make_listing on one uploaded file and append a photo record.

    make_listing writes the subject arrays into session_state; capture them
    immediately after the call so every photo keeps its own copy.
    Returns a warning string on failure, None on success.
    """
    try:
        f.seek(0)
        pil_img = ImageOps.exif_transpose(Image.open(f))
    except Exception as exc:
        return f"{getattr(f, 'name', 'image')}: could not open ({exc})."
    result, warn = make_listing(pil_img)
    if warn:
        return f"{getattr(f, 'name', 'image')}: {warn}"
    st.session_state.photos.append(_new_photo(
        getattr(f, "name", "photo"), pil_img, result,
        st.session_state.subject_rgba, st.session_state.subject_mask,
    ))
    return None


def _process_files(files) -> None:
    warnings = [w for w in (_process_one(f) for f in files) if w]
    st.session_state.warn = "  ".join(warnings) if warnings else None

# ── Nav pill + view router ─────────────────────────────────────────────────────
# Both tools run on this single Streamlit server, so the app works when
# deployed (e.g. Streamlit Cloud) where only one port is exposed.
# ?view=social renders the IAAC app inline via st.components.v1.html.
_NAV_CSS = """\
#cj-nav{display:flex;justify-content:space-between;align-items:center;
        padding-bottom:1rem;margin-bottom:1.8rem;border-bottom:1px solid #cfc4c5}
#cj-nav .cj-brand{font-family:'Libre Caslon Text',Georgia,serif;font-size:1.55rem;
        color:#1b1b1b;letter-spacing:-0.01em;line-height:1}
#cj-nav .cj-links{display:inline-flex;gap:1.6rem;align-items:center}
#cj-nav .cj-np{font-family:'Hanken Grotesk',sans-serif;font-size:.78rem;
  font-weight:600;letter-spacing:.06em;text-transform:uppercase;cursor:default;
  color:#5d5e66 !important;text-decoration:none !important;display:inline-block;
  padding-bottom:3px;border-bottom:2px solid transparent;
  transition:color .15s,border-color .15s;line-height:1.4}
#cj-nav .cj-np.active{color:#1b1b1b !important;border-bottom-color:#1b1b1b}
#cj-nav a.cj-np{cursor:pointer}
#cj-nav a.cj-np:hover{color:#1b1b1b !important}"""

_IAAC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "iaac")


@st.cache_resource(show_spinner=False)
def _iaac_component_html() -> str:
    """Build a self-contained HTML document for the IAAC tool.

    Streamlit's static file server labels text files as text/plain + nosniff,
    so <link>/<script src> tags pointing at it are blocked by the browser.
    CSS and JS are therefore inlined.  Binary assets (shape PNGs, woff fonts)
    are fine — they load through <base href> from /app/static/iaac/
    (enableStaticServing = true in .streamlit/config.toml).  Pattern SVGs
    would be blocked too, so they're injected as base64 data URIs and the
    bundle's URL builder is patched to use them.
    """
    with open(os.path.join(_IAAC_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(_IAAC_DIR, "bundle", "main.css"), encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(_IAAC_DIR, "bundle", "main.js"), encoding="utf-8") as f:
        js = f.read()

    # Font urls in the bundle are relative to bundle/; the document base
    # points at the iaac root, so drop the ../
    css = css.replace('url("../assets/', 'url("assets/')

    patterns = {}
    pattern_dir = os.path.join(_IAAC_DIR, "assets", "patterns")
    for name in sorted(os.listdir(pattern_dir)):
        if name.startswith("pattern-") and name.endswith(".svg") and " " not in name:
            key = name[len("pattern-"):-len(".svg")]
            with open(os.path.join(pattern_dir, name), "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            patterns[key] = f"data:image/svg+xml;base64,{b64}"

    js = js.replace(
        "return `assets/patterns/pattern-${value}.svg`;",
        "return (window.__IAAC_PATTERNS||{})[value]"
        " || `assets/patterns/pattern-${value}.svg`;",
    )

    html = html.replace("<head>", '<head>\n\t<base href="/app/static/iaac/">', 1)
    html = html.replace(
        '<link rel="stylesheet" href="bundle/main.css"/>',
        "<style>html,body{height:100%}</style>\n\t<style>\n" + css + "\n\t</style>",
    )
    html = html.replace(
        '<script src="bundle/main.js"></script>',
        "<script>window.__IAAC_PATTERNS=" + json.dumps(patterns) + "</script>\n"
        "\t\t<script>\n" + js + "\n\t\t</script>",
    )
    return html


def _nav_pill(active: str) -> str:
    listing = (
        '<span class="cj-np active">Listing</span>' if active == "listing"
        else '<a class="cj-np" href="?view=listing" target="_self">Listing</a>'
    )
    social = (
        '<span class="cj-np active">Social</span>' if active == "social"
        else '<a class="cj-np" href="?view=social" target="_self">Social</a>'
    )
    return (
        '<div id="cj-nav">'
        '<span class="cj-brand">CJ&nbsp;Studio</span>'
        f'<div class="cj-links">{listing}{social}</div>'
        '</div>'
    )


# ── Lumina-style app shell (fixed left sidebar) ────────────────────────────────
_SHELL_CSS = """\
.material-symbols-outlined{font-family:'Material Symbols Outlined';font-weight:normal;
  font-style:normal;line-height:1;letter-spacing:normal;text-transform:none;
  display:inline-block;white-space:nowrap;direction:ltr;
  font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24}
/* push the main column clear of the fixed sidebar */
.block-container{max-width:1180px !important;padding-left:280px !important}
#cj-side{position:fixed;left:0;top:0;bottom:0;width:256px;z-index:50;
  background:#f3f4f5;border-right:1px solid #cfc4c5;
  display:flex;flex-direction:column;padding:16px;
  font-family:'Hanken Grotesk',sans-serif}
#cj-side .cj-brand{font-family:'Libre Caslon Text',Georgia,serif;font-size:1.55rem;
  color:#1b1b1b;letter-spacing:-0.01em;line-height:1.1;padding:8px 16px}
#cj-side .cj-brand small{display:block;font-family:'Hanken Grotesk',sans-serif;
  font-size:.58rem;font-weight:300;letter-spacing:.2em;text-transform:uppercase;
  color:#5d5e66;margin-top:8px}
#cj-side .cj-navcol{display:flex;flex-direction:column;gap:4px;margin-top:32px;flex-grow:1}
#cj-side a.cj-item{display:flex;align-items:center;gap:12px;padding:8px 16px;
  border-radius:8px;text-decoration:none !important;font-size:.85rem;
  color:#5d5e66 !important;font-weight:300;transition:background .15s,color .15s}
#cj-side a.cj-item .material-symbols-outlined{font-size:20px}
#cj-side a.cj-item:hover{background:#e7e8e9;color:#1b1b1b !important}
#cj-side a.cj-item.active{background:#1b1b1b;color:#fff !important;font-weight:400}
#cj-mobtop{display:none;align-items:center;padding-bottom:1rem;margin-bottom:1.2rem;
  border-bottom:1px solid #cfc4c5}
#cj-mobtop .cj-brand{font-family:'Libre Caslon Text',Georgia,serif;font-size:1.4rem;color:#1b1b1b}
@media (max-width:900px){
  #cj-side{display:none}
  .block-container{padding-left:1rem !important}
  #cj-mobtop{display:flex}
}"""

# ── Home (upload) screen extras: drop zone, how-it-works strip, footer ─────────
_HOME_EXTRA_CSS = """\
.cj-home-top{height:3.5rem}
[data-testid="stFileUploader"]{border:none !important;background:transparent !important;
  padding:0 !important;margin:0 !important}
[data-testid="stFileUploader"] > label{display:none}
section[data-testid="stFileUploaderDropzone"]{position:relative;min-height:380px;
  padding:0 !important;border:2px dashed #cfc4c5;border-radius:0;
  background:rgba(255,255,255,.5);display:flex !important;flex-direction:column !important;
  align-items:center !important;justify-content:center !important;overflow:hidden;
  transition:border-color .3s ease,background-color .3s ease}
section[data-testid="stFileUploaderDropzone"]:hover{border-color:#000;background:rgba(0,0,0,.02)}
section[data-testid="stFileUploaderDropzone"]::before{content:"";position:absolute;inset:0;
  pointer-events:none;opacity:.025;background-image:radial-gradient(#000 1px,transparent 1px);
  background-size:24px 24px}
section[data-testid="stFileUploaderDropzone"]::after{content:"JPG \\00B7 PNG \\00B7 WEBP  /  MAX 200MB";
  font-family:'Hanken Grotesk',sans-serif;font-size:.72rem;letter-spacing:.12em;
  text-transform:uppercase;color:#5d5e66;font-weight:300;margin-top:14px}
[data-testid="stFileUploaderDropzoneInstructions"]{display:flex !important;
  flex-direction:column !important;align-items:center !important;gap:0;width:100%;z-index:1}
[data-testid="stFileUploaderDropzoneInstructions"] > *{display:none !important}
[data-testid="stFileUploaderDropzoneInstructions"]::before{font-family:'Material Symbols Outlined';
  content:"\\e43e";font-size:40px;color:#1b1b1b;width:80px;height:80px;border-radius:50%;
  background:#edeeef;display:flex;align-items:center;justify-content:center;margin-bottom:24px;
  transition:background .3s,color .3s}
[data-testid="stFileUploaderDropzoneInstructions"]::after{content:"Click or drop files";
  font-family:'Libre Caslon Text',Georgia,serif;font-size:1.8rem;color:#1b1b1b}
section[data-testid="stFileUploaderDropzone"]:hover [data-testid="stFileUploaderDropzoneInstructions"]::before{
  background:#1b1b1b;color:#fff}
section[data-testid="stFileUploaderDropzone"] > button{position:absolute !important;
  inset:0;width:100% !important;height:100% !important;margin:0 !important;padding:0 !important;
  border:none !important;background:transparent !important;color:transparent !important;
  opacity:0;cursor:pointer;z-index:3;box-shadow:none !important}
.cj-strip{display:flex;justify-content:center;align-items:center;gap:2.5rem;flex-wrap:wrap;
  border-top:1px solid #cfc4c5;margin-top:3rem;padding-top:1.6rem;opacity:.6;transition:opacity .3s}
.cj-strip:hover{opacity:1}
.cj-step{display:flex;align-items:center;gap:.75rem}
.cj-num{width:24px;height:24px;border-radius:50%;background:#e7e8e9;color:#1b1b1b;
  display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600}
.cj-steplbl{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:#5d5e66;font-weight:300}
.cj-dash{color:#cfc4c5}"""

# ── Result screen: canvas, before→after reveal animation, control panel ────────
_RESULT_CSS = """\
/* ===== main area fills exactly the viewport, no page scroll ===== */
[data-testid="stHeader"]{display:none !important}
[data-testid="stMain"]{height:100vh !important;overflow:hidden !important}
.block-container{max-width:100% !important;padding:0 !important;
  height:100vh !important;overflow:hidden !important}
[data-testid="stMainBlockContainer"]{padding:0 !important;
  height:100vh !important;overflow:hidden !important}
/* kill the inter-element gap that pushes the canvas row down from the top */
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlockBorderWrapper"]
  >[data-testid="stVerticalBlock"]{gap:0 !important;height:100vh !important}
@ROW@{gap:0 !important;align-items:stretch !important;height:100vh !important;
  min-height:0;margin-left:256px;flex-wrap:nowrap !important}
/* panel scrolls internally if its own content ever exceeds the viewport */
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]{overflow-y:auto}
/* flex-basis 0 (not auto): a wide filmstrip must never set the column's size,
   otherwise the row wraps and the panel drops below the canvas */
@CANVAS@{flex:1 1 0% !important;width:auto !important;min-width:0}
@PANEL@{flex:0 0 384px !important;width:384px !important;min-width:384px !important}
/* canvas column stretches to the row (align-items:stretch); the inner chain
   fills that height. NB: don't set height on the column itself — a resolved
   percentage height there cancels the stretch and collapses it. */
@CANVAS@ [data-testid="stVerticalBlockBorderWrapper"],
@CANVAS@ [data-testid="stVerticalBlock"]{height:100% !important;min-height:0}
/* cj_canvas is a flex column: stage (before/after, flex) + filmstrip (auto) */
@CANVAS@ .st-key-cj_canvas{display:flex !important;flex-direction:column !important;gap:0 !important}
@CANVAS@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_stage){
  flex:1 1 auto !important;min-height:0}
@CANVAS@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_film){
  flex:0 0 auto !important;height:auto !important}
@CANVAS@ .st-key-cj_stage{height:100% !important;min-height:0;position:relative}
@CANVAS@ .st-key-cj_stage [data-testid="stElementContainer"]{height:auto !important}
@CANVAS@ .st-key-cj_stage [data-testid="stElementContainer"]:has(iframe){height:100% !important;min-height:0}
@CANVAS@ iframe{height:100% !important;border:none;display:block;width:100%}
/* ===== edit panel (RIGHT) — Figma 182:544 / 182:605, pixel-exact, nested ===== */
/* height chain: column (stretched to the row) → card, so it fills the viewport */
@PANEL@ > [data-testid="stVerticalBlockBorderWrapper"],
@PANEL@ > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"]{
  height:100% !important;min-height:0}
/* aside card = cj_card (the ONLY styled wrapper); strip its 15px padding */
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_card){
  height:100%;border:none !important;border-left:1px solid #cfc4c5 !important;
  border-radius:0 !important;background:#fff;padding:0 !important;overflow:hidden}
.st-key-cj_card{height:100%;display:flex;flex-direction:column;gap:0 !important;padding:0 !important}
/* the "Adjust" header markdown has a -16px bottom margin by default — cancel it */
@PANEL@ [data-testid="stElementContainer"]:has(.cj-panel-head) [data-testid="stMarkdownContainer"]{margin:0 !important}
/* nested containers (body/groups/desc): strip Streamlit's border / bg / padding */
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_body),
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_grp1),
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_grp2),
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_desc){
  border:none !important;background:transparent !important;padding:0 !important}
/* body fills height; pad 48 all sides; gap 48 between the two groups */
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_body){flex:1 1 auto !important;
  min-height:0;overflow:auto}
.st-key-cj_body{padding:48px !important;gap:48px !important}
/* groups: no padding, gap 16 */
.st-key-cj_grp1,.st-key-cj_grp2{gap:16px !important;padding:0 !important}
/* cancel the markdown's default -16px bottom margin so the 16px gap shows */
.st-key-cj_grp1 [data-testid="stMarkdownContainer"]{margin:0 !important}
/* header: pad 48/48/53, border-b; "Adjust" Libre Caslon 48/56 tracking -0.96px */
.cj-panel-head{padding:48px;border-bottom:1px solid #cfc4c5}
.cj-panel-head h3{font-family:'Libre Caslon Text',Georgia,serif;font-size:48px;line-height:56px;
  letter-spacing:-0.96px;font-weight:400;color:#000;margin:0 !important;padding:0 !important}
/* section headers: Libre Caslon 24/32 black (Output Ratio + Text Overlay) */
.cj-sec,.st-key-cj_txt [data-testid="stWidgetLabel"] p,.st-key-cj_txt [data-testid="stWidgetLabel"]{
  font-family:'Libre Caslon Text',Georgia,serif !important;font-size:24px !important;
  line-height:32px !important;font-weight:400 !important;color:#000 !important;margin:0}
/* output-ratio row: gap-8; cards py-25 px-1 rounded-2 */
.st-key-cj_grp1 [data-testid="stHorizontalBlock"]{gap:8px !important}
[class*="st-key-cjr_"] button{display:flex !important;flex-direction:column;align-items:center;
  justify-content:center;gap:8px !important;padding:25px 1px !important;border-radius:0 !important;
  font-family:'Libre Caslon Text',Georgia,serif !important;font-size:24px !important;line-height:32px !important;
  font-weight:400 !important;letter-spacing:0 !important;transition:border-color .15s,color .15s}
[class*="st-key-cjr_"] button[kind="secondary"]{background:#f8f9fa !important;color:#5d5e66 !important;
  border:1px solid #cfc4c5 !important}
[class*="st-key-cjr_"] button[kind="secondary"]:hover{border-color:#7e7576 !important;color:#000 !important}
[class*="st-key-cjr_"] button[kind="primary"]{background:#000 !important;color:#fff !important;
  border:1px solid #000 !important}
[class*="st-key-cjr_"] button::after{font-family:'Hanken Grotesk',sans-serif;font-size:12px;
  line-height:16px;font-weight:500;letter-spacing:1.2px;text-transform:uppercase}
[class*="st-key-cjr_"] button[kind="primary"]::after{opacity:.8}
.st-key-cjr_1x1 button::after{content:"EBAY"}
.st-key-cjr_4x3 button::after{content:"WEBSITE"}
.st-key-cjr_4x5 button::after{content:"INSTAGRAM"}
/* text-overlay row: h3 left / switch right; ONLY the switch is clickable */
.st-key-cj_txt label{flex-direction:row-reverse;justify-content:space-between;width:100%;
  align-items:center;gap:12px;pointer-events:none}
.st-key-cj_txt [data-testid="stWidgetLabel"]{flex:1 1 auto !important;min-width:0;white-space:nowrap}
/* toggle track = the FIRST div child only (the 2nd div holds the label text) */
.st-key-cj_txt label>div:first-of-type{width:48px !important;height:24px !important;flex:0 0 48px !important;
  border-radius:12px !important;background:#e7e8e9 !important;position:relative;border:none !important;
  pointer-events:auto;cursor:pointer}
.st-key-cj_txt label>div:first-of-type>div{width:20px !important;height:20px !important;border-radius:12px !important;
  background:#7e7576 !important;position:absolute !important;top:2px !important;left:2px !important;
  transform:none !important;box-shadow:none !important;transition:transform .3s,background .3s !important}
.st-key-cj_txt label:has(input:checked)>div:first-of-type{background:#000 !important}
.st-key-cj_txt label:has(input:checked)>div:first-of-type>div{background:#e7e8e9 !important;transform:translateX(24px) !important}
/* description box reveals; group2 gap-16 provides the spacing above it.
   .st-key-cj_desc IS the stVerticalBlock, so put the flex row directly on it:
   input grows, Apply button sits 4px to its right, both stretched to equal height. */
.st-key-cj_desc{overflow:hidden;max-height:0;opacity:0;
  display:flex !important;flex-direction:row !important;gap:8px !important;
  align-items:stretch !important;
  transition:max-height .5s ease-in-out,opacity .5s ease-in-out}
.st-key-cj_desc>[data-testid="stElementContainer"]:first-child{flex:1 1 auto;min-width:0}
.st-key-cj_grp2:has(.st-key-cj_txt input:checked) .st-key-cj_desc{max-height:120px;opacity:1}
/* filled input box: single 1px #cfc4c5 border on the OUTER root only (matches the
   Output Ratio buttons); baseweb nests two divs, so clear the inner one's border to
   avoid a doubled line. bg #f8f9fa, r-2; text Hanken 18, placeholder #5d5e66@50% */
.st-key-cj_desc [data-baseweb="input"]{
  background:#f8f9fa !important;border:1px solid #cfc4c5 !important;border-radius:0 !important}
.st-key-cj_desc [data-baseweb="input"]:focus-within{border:1px solid #cfc4c5 !important}
.st-key-cj_desc [data-baseweb="base-input"]{
  background:transparent !important;border:none !important;border-radius:0 !important}
.st-key-cj_desc input{font-family:'Hanken Grotesk',sans-serif !important;font-size:18px !important;
  line-height:normal !important;color:#191c1d !important;background:transparent !important;padding:8px 12px !important}
.st-key-cj_desc input::placeholder{color:rgba(93,94,102,0.5) !important}
/* Apply button: own column right of the input, equal height, hidden while empty.
   Stretch the whole chain (container → stButton wrapper → button) so the black
   button fills the row height instead of sitting 25px tall in the middle. */
.st-key-cj_apply{flex:0 0 auto !important;width:auto !important;margin:0 !important;
  display:flex !important;align-items:stretch !important}
.st-key-cj_apply [data-testid="stButton"]{width:auto !important;margin:0 !important;
  display:flex !important;align-items:stretch !important}
.st-key-cj_apply button{align-self:stretch !important;height:auto !important;min-height:0 !important;
  display:flex !important;align-items:center !important;justify-content:center !important;
  padding:0 18px !important;border:none !important;background:#191c1d !important;border-radius:0 !important;
  font-family:'Hanken Grotesk',sans-serif !important;font-size:13px !important;font-weight:600 !important;
  letter-spacing:1px !important;text-transform:uppercase !important;color:#fff !important}
.st-key-cj_apply button>*{margin:0 !important}
.st-key-cj_apply button:hover{opacity:.9 !important;background:#191c1d !important;color:#fff !important}
.st-key-cj_desc:has(input:placeholder-shown) .st-key-cj_apply{display:none !important}
/* footer: pad 49/48/48 border-t; Export py-16 black; Try py-17 border #7e7576; 14/20 semibold 1.4px */
.cj-foot{padding:48px;border-top:1px solid #cfc4c5;display:flex;flex-direction:column;gap:16px}
.cj-btn-primary,.cj-btn-outline{display:block;text-align:center;text-decoration:none !important;
  font-family:'Hanken Grotesk',sans-serif;font-size:14px;line-height:20px;font-weight:600;
  letter-spacing:1.4px;text-transform:uppercase;cursor:pointer;border-radius:0}
.cj-btn-primary,.cj-btn-primary:link,.cj-btn-primary:visited{background:#000;color:#fff !important;padding:16px}
.cj-btn-primary:hover{opacity:.9;color:#fff !important}
.cj-btn-outline,.cj-btn-outline:link,.cj-btn-outline:visited{background:transparent;
  color:#000 !important;border:1px solid #7e7576;padding:17px 1px}
.cj-btn-outline:hover{background:#f8f9fa;color:#000 !important}
/* ===== loading overlay: shows over the canvas whenever a rerun is running ===== */
/* Streamlit makes every stElementContainer position:relative, so the overlay would
   anchor to its own (below-the-iframe) container. Pull that container out of flow and
   pin it over cj_stage so the overlay lands exactly on the image. */
.st-key-cj_stage>[data-testid="stElementContainer"]:has(.cj-loading-overlay){
  position:absolute !important;inset:0 !important;margin:0 !important;height:auto !important;
  z-index:30;pointer-events:none !important}
/* only capture pointer events while a rerun is running, so the idle slider/buttons work */
[data-test-script-state="running"] .st-key-cj_stage>[data-testid="stElementContainer"]:has(.cj-loading-overlay){
  pointer-events:auto !important}
.cj-loading-overlay{position:absolute;inset:0;z-index:30;display:flex;
  align-items:center;justify-content:center;background:rgba(255,255,255,.2);
  opacity:0;visibility:hidden;pointer-events:none;transition:opacity .12s ease}
[data-testid="stApp"][data-test-script-state="running"] .cj-loading-overlay,
[data-testid="stApp"][data-test-script-state="rerunRequested"] .cj-loading-overlay{
  opacity:1 !important;visibility:visible !important;pointer-events:auto}
/* keep the canvas + overlay containers un-dimmed while Streamlit marks them stale */
[data-test-script-state="running"] .st-key-cj_stage>[data-testid="stElementContainer"]{
  opacity:1 !important}
/* frying-pan + egg loader (user-provided) */
.loader{--color-1:#3e494d;--color-2:#5d6063;--color-3:#6c4924;--color-4:#4b2d21;
  --color-5:#4d5457;--color-6:#9f9e9e;--color-7:#fff;--color-8:#fff6;--color-9:#ffc400;
  --color-10:#ffae00;--color-11:#0002;--color-12:#0003;--size:1.4px;
  position:relative;width:calc(120*var(--size));height:calc(14*var(--size));
  border-radius:0 0 calc(15*var(--size)) calc(15*var(--size));background-color:var(--color-1);
  box-shadow:0 calc(-1*var(--size)) calc(4*var(--size)) var(--color-2) inset;
  animation:panex .5s linear alternate infinite;transform-origin:calc(170*var(--size)) 0;
  z-index:10;perspective:calc(300*var(--size))}
.loader::before{content:'';position:absolute;left:calc(100% - calc(2*var(--size)));top:0;z-index:-2;
  height:calc(10*var(--size));width:calc(70*var(--size));
  border-radius:0 calc(4*var(--size)) calc(4*var(--size)) 0;background-repeat:no-repeat;
  background-image:linear-gradient(var(--color-3),var(--color-4)),
    linear-gradient(var(--color-5) calc(24*var(--size)),transparent 0),
    linear-gradient(var(--color-6) calc(24*var(--size)),transparent 0);
  background-size:calc(50*var(--size)) calc(10*var(--size)),calc(4*var(--size)) calc(8*var(--size)),
    calc(24*var(--size)) calc(4*var(--size));
  background-position:right center,calc(17*var(--size)) center,0 center}
.loader::after{content:'';position:absolute;left:50%;top:0;z-index:-2;
  transform:translate(-50%,calc(-20*var(--size))) rotate3d(75,-2,3,78deg);
  width:calc(55*var(--size));height:calc(53*var(--size));background:var(--color-7);
  background-image:radial-gradient(circle calc(3*var(--size)),var(--color-8) 90%,transparent 10%),
    radial-gradient(circle calc(12*var(--size)),var(--color-9) 90%,transparent 10%),
    radial-gradient(circle calc(12*var(--size)),var(--color-10) 100%,transparent 0);
  background-repeat:no-repeat;
  background-position:calc(-4*var(--size)) calc(-6*var(--size)),calc(-2*var(--size)) calc(-2*var(--size)),
    calc(-1*var(--size)) calc(-1*var(--size));
  box-shadow:calc(-2*var(--size)) calc(-3*var(--size)) var(--color-11) inset,
    0 0 calc(4*var(--size)) var(--color-12) inset;
  border-radius:47% 36% 50% 50%/49% 45% 42% 44%;animation:eggRst 1s ease-out infinite}
@keyframes eggRst{0%,100%{transform:translate(-50%,calc(-20*var(--size))) rotate3d(90,0,0,90deg);opacity:0}
  10%,90%{transform:translate(-50%,calc(-30*var(--size))) rotate3d(90,0,0,90deg);opacity:1}
  25%{transform:translate(-50%,calc(-40*var(--size))) rotate3d(85,17,2,70deg)}
  75%{transform:translate(-50%,calc(-40*var(--size))) rotate3d(75,-3,2,70deg)}
  50%{transform:translate(-55%,calc(-50*var(--size))) rotate3d(75,-8,3,50deg)}}
@keyframes panex{0%{transform:rotate(-5deg)}100%{transform:rotate(10deg)}}
@media (max-width:900px){
  [data-testid="stMain"],.block-container,[data-testid="stMainBlockContainer"],
  [data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlockBorderWrapper"]
    >[data-testid="stVerticalBlock"]{height:auto !important;overflow:visible !important}
  @ROW@{margin-left:0;height:auto !important;min-height:0;flex-wrap:wrap}
  @CANVAS@,@PANEL@{flex:1 1 100% !important;width:100% !important}
  @CANVAS@ iframe{height:70vh !important}
}""".replace(
    "@ROW@", '[data-testid="stHorizontalBlock"]:has(.cj-panel-head)'
).replace(
    "@CANVAS@",
    '[data-testid="stHorizontalBlock"]:has(.cj-panel-head)>[data-testid="stColumn"]:first-child',
).replace(
    "@PANEL@",
    '[data-testid="stHorizontalBlock"]:has(.cj-panel-head)>[data-testid="stColumn"]:last-child',
)

# ── Multi-photo extras: filmstrip, add-more tile, footer buttons ───────────────
_MULTI_CSS = """\
/* filmstrip bar under the before/after view — fixed to the column width; when
   the tiles overflow it scrolls horizontally instead of growing the layout */
[data-testid="stVerticalBlockBorderWrapper"]:has(>.st-key-cj_film){
  flex:0 0 auto !important;height:auto !important;min-width:0 !important;
  max-width:100% !important;overflow:hidden}
.st-key-cj_film{border-top:1px solid #cfc4c5;background:#f8f9fa;
  padding:16px 24px !important;display:flex !important;flex-direction:row !important;
  flex-wrap:nowrap !important;gap:12px !important;align-items:center !important;
  overflow-x:auto !important;height:auto !important;
  width:100% !important;min-width:0 !important;max-width:100% !important}
.st-key-cj_film::-webkit-scrollbar{height:6px}
.st-key-cj_film::-webkit-scrollbar-track{background:transparent}
.st-key-cj_film::-webkit-scrollbar-thumb{background:#cfc4c5;border-radius:3px}
.st-key-cj_film>[data-testid="stElementContainer"],
.st-key-cj_film>[data-testid="stVerticalBlockBorderWrapper"]{width:auto !important;
  flex:0 0 auto !important;height:auto !important;margin:0 !important;min-width:0}
/* per-photo wrapper: relative anchor for the hover delete button */
[class*="st-key-cj_tw_"]{position:relative;gap:0 !important;padding:0 !important}
[class*="st-key-cj_tw_"]>[data-testid="stElementContainer"]{margin:0 !important;width:auto !important}
/* thumbnails = square buttons painted with the enhanced result; hover matches
   the Output Ratio cards (border darkens to #7e7576) */
[class*="st-key-cj_thumb_"] button{width:72px !important;height:72px !important;
  min-height:72px !important;padding:0 !important;border-radius:2px !important;
  border:1px solid #cfc4c5 !important;background-size:cover !important;
  background-position:center !important;background-repeat:no-repeat !important;
  color:transparent !important;filter:grayscale(1);opacity:.65;
  transition:opacity .15s,filter .15s,border-color .15s}
[class*="st-key-cj_tw_"]:hover [class*="st-key-cj_thumb_"] button{
  opacity:1;filter:none;border-color:#7e7576 !important}
[class*="st-key-cj_thumb_"] button p{display:none}
/* delete affordance: ✕ chip in the tile's top-right corner, shown on hover;
   clicking it opens the confirm dialog */
[class*="st-key-cj_delbtn_"]{position:absolute !important;top:3px;right:3px;z-index:6;
  margin:0 !important;width:auto !important;opacity:0;pointer-events:none;
  transition:opacity .15s}
[class*="st-key-cj_tw_"]:hover [class*="st-key-cj_delbtn_"]{opacity:1;pointer-events:auto}
[class*="st-key-cj_delbtn_"] button{width:20px !important;height:20px !important;
  min-height:20px !important;padding:0 !important;border-radius:50% !important;
  border:1px solid #cfc4c5 !important;background:#fff !important;color:#1b1b1b !important;
  display:flex !important;align-items:center !important;justify-content:center !important;
  box-shadow:0 1px 3px rgba(0,0,0,.25);line-height:1 !important}
[class*="st-key-cj_delbtn_"] button:hover{background:#1b1b1b !important;color:#fff !important}
[class*="st-key-cj_delbtn_"] button p{font-size:11px !important;line-height:1 !important;
  font-weight:600 !important}
/* pending tiles: freshly added photos processing in place, spinner overlay */
.cj-pend-row{display:flex;gap:12px}
.cj-pend{position:relative;width:72px;height:72px;flex:0 0 auto;
  border:1px solid #cfc4c5;border-radius:2px;background:#e7e8e9;
  background-size:cover;background-position:center}
.cj-pend.dim{opacity:.5}
.cj-pend.spin::before{content:"";position:absolute;inset:0;background:rgba(255,255,255,.5)}
.cj-pend.spin::after{content:"";position:absolute;top:50%;left:50%;width:18px;height:18px;
  margin:-10px 0 0 -10px;border:2px solid #cfc4c5;border-top-color:#1b1b1b;
  border-radius:50%;animation:cjspin .8s linear infinite}
@keyframes cjspin{to{transform:rotate(360deg)}}
/* add-more tile = compact uploader styled as a dashed "+" square */
[class*="st-key-cj_more_"]{width:auto !important;flex:0 0 auto !important;margin:0 !important}
[class*="st-key-cj_more_"] [data-testid="stFileUploader"]{border:none !important;
  background:transparent !important;padding:0 !important;margin:0 !important;width:72px}
[class*="st-key-cj_more_"] [data-testid="stFileUploader"]>label{display:none}
[class*="st-key-cj_more_"] section[data-testid="stFileUploaderDropzone"]{
  width:72px;height:72px;min-height:72px;padding:0 !important;border:1.5px dashed #cfc4c5;
  border-radius:2px;background:#fff;display:flex !important;align-items:center !important;
  justify-content:center !important;overflow:hidden;transition:border-color .15s}
[class*="st-key-cj_more_"] section[data-testid="stFileUploaderDropzone"]:hover{border-color:#1b1b1b}
[class*="st-key-cj_more_"] [data-testid="stFileUploaderDropzoneInstructions"]{margin:0 !important;
  display:flex !important;flex-direction:column !important;align-items:center !important}
[class*="st-key-cj_more_"] [data-testid="stFileUploaderDropzoneInstructions"]>*{display:none !important}
[class*="st-key-cj_more_"] [data-testid="stFileUploaderDropzoneInstructions"]::before{
  content:"+";font-family:'Hanken Grotesk',sans-serif;font-size:32px;font-weight:300;
  line-height:1;color:#5d5e66}
[class*="st-key-cj_more_"] section[data-testid="stFileUploaderDropzone"]>button{
  position:absolute !important;inset:0;width:100% !important;height:100% !important;
  margin:0 !important;padding:0 !important;border:none !important;background:transparent !important;
  color:transparent !important;opacity:0;cursor:pointer}
/* the uploader's own file chips would stack under the + tile — pending tiles
   in the strip replace them, so hide the chips entirely */
[class*="st-key-cj_more_"] [data-testid="stFileUploaderFile"],
[class*="st-key-cj_more_"] [data-testid="stFileUploaderPagination"],
[class*="st-key-cj_more_"] [data-testid="stFileUploaderDeleteBtn"]{display:none !important}
/* footer buttons (replace the old HTML anchors) */
.st-key-cj_foot{padding:48px;border-top:1px solid #cfc4c5;display:flex !important;
  flex-direction:column !important;gap:16px !important}
.st-key-cj_foot [data-testid="stElementContainer"]{width:100% !important;margin:0 !important}
.st-key-cj_foot button,.st-key-cj_foot [data-testid="stDownloadButton"] button{
  width:100% !important;border-radius:0 !important;font-family:'Hanken Grotesk',sans-serif !important;
  font-size:14px !important;font-weight:600 !important;letter-spacing:1.4px !important;
  text-transform:uppercase !important;padding:16px !important;border:none !important}
.st-key-cj_export_all button,.st-key-cj_export_one button{background:#000 !important;color:#fff !important}
.st-key-cj_export_all button:hover,.st-key-cj_export_one button:hover{opacity:.9;background:#000 !important;color:#fff !important}
.st-key-cj_tryanother button{background:transparent !important;color:#000 !important;
  border:1px solid #7e7576 !important;padding:17px 1px !important}
.st-key-cj_tryanother button:hover{background:#f8f9fa !important;color:#000 !important}
.st-key-cj_foot button p{font-size:14px !important;font-weight:600 !important;letter-spacing:1.4px !important}
"""

# ── Export selection modal styling (Figma 233:833 grid) ────────────────────────
_EXPORT_CSS = """\
[class*="st-key-cj_ex_"]{position:relative}
.cj-ex-tile{position:relative;aspect-ratio:1/1;border-radius:2px;overflow:hidden;
  border:1px solid #cfc4c5;background:#e1e3e4}
.cj-ex-tile.sel{border:2px solid #000}
.cj-ex-tile img{width:100%;height:100%;object-fit:cover;opacity:.7;transition:opacity .15s}
.cj-ex-tile.sel img{opacity:1}
.cj-ex-check{position:absolute;top:8px;right:8px;width:20px;height:20px;
  border-radius:50%;background:rgba(248,249,250,.7);border:1.5px solid #7e7576;
  box-sizing:border-box}
.cj-ex-check.on{background:#000;border-color:#000}
.cj-ex-check.on::after{content:"";position:absolute;left:6px;top:2.5px;
  width:5px;height:10px;border:solid #fff;border-width:0 2px 2px 0;
  transform:rotate(45deg)}
/* invisible pick button stretched over each tile */
[class*="st-key-cj_exbtn_"]{position:absolute !important;inset:0 !important;z-index:5;margin:0 !important}
[class*="st-key-cj_exbtn_"] button{width:100% !important;height:100% !important;
  opacity:0 !important;cursor:pointer;background:transparent !important;border:none !important;
  min-height:0 !important}
.st-key-cj_ex_actions [data-testid="stButton"] button,
.st-key-cj_ex_actions [data-testid="stDownloadButton"] button{width:100% !important;
  border-radius:0 !important;text-transform:uppercase;letter-spacing:1.2px;font-weight:600}
.st-key-cj_ex_go button{background:#000 !important;color:#fff !important;border:none !important}
.st-key-cj_ex_go button:hover{opacity:.9;background:#000 !important;color:#fff !important}
"""


def _sidebar_html(active: str = "upload") -> str:
    def _item(href: str, icon: str, label: str, is_active: bool) -> str:
        cls = "cj-item active" if is_active else "cj-item"
        fill = "font-variation-settings:'FILL' 1;" if is_active else ""
        return (
            f'<a class="{cls}" href="{href}" target="_self">'
            f'<span class="material-symbols-outlined" style="{fill}">{icon}</span>'
            f'<span>{label}</span></a>'
        )

    nav = (
        _item("?view=listing", "upload_file", "Upload", active == "upload")
        + _item("?view=social", "grid_view", "Social", active == "social")
        + '<a class="cj-item" href="#"><span class="material-symbols-outlined">photo_library</span>'
          '<span>Library</span></a>'
    )
    settings = (
        '<a class="cj-item" href="#"><span class="material-symbols-outlined">settings</span>'
        '<span>Settings</span></a>'
    )
    return (
        '<div id="cj-mobtop"><span class="cj-brand">CJ&nbsp;Studio</span></div>'
        '<div id="cj-side">'
        '<div class="cj-brand">CJ&nbsp;Studio<small>Internal Tools</small></div>'
        f'<div class="cj-navcol">{nav}</div>'
        f'<div class="cj-navfoot">{settings}</div>'
        '</div>'
    )


# ── In-place processing card (replaces the drop zone while photos process) ─────
# Square corners throughout (border-radius 0); one row per uploaded image with
# its own progress line; the batch header shows overall progress. The row list
# height is capped so many images scroll inside the card instead of growing it.
_PROC_CSS = """\
.cj-proc{position:relative;border:1px solid #cfc4c5;border-radius:0;background:#fff;
  min-height:380px;display:flex;flex-direction:column;justify-content:center;gap:16px;
  padding:1.5rem 1.5rem 3.2rem}
.cj-proc-cancel,.cj-proc-cancel:link,.cj-proc-cancel:visited{position:absolute;left:0;right:0;
  bottom:20px;text-align:center;font-family:'Hanken Grotesk',sans-serif;font-size:12px;
  color:#9a9a9a !important;text-decoration:underline !important;text-underline-offset:2px}
.cj-proc-cancel:hover{color:#5d5e66 !important}
.cj-proc-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  padding-bottom:12px;border-bottom:1px solid #e7e8e9}
.cj-proc-head-label{font-family:'Hanken Grotesk',sans-serif;font-size:14px;font-weight:600;
  color:#1b1b1b}
.cj-proc-head-pct{font-family:'Hanken Grotesk',sans-serif;font-size:12px;color:#5d5e66}
.cj-proc-head-track{width:100%;height:2px;background:#e1e3e4;position:relative;
  overflow:hidden;margin-top:10px}
.cj-proc-head-fill{position:absolute;top:0;left:0;height:100%;background:#1b1b1b;
  transition:width .4s ease}
.cj-proc-list{display:flex;flex-direction:column;gap:12px;overflow-y:auto;
  max-height:300px;padding-right:4px}
.cj-proc-row{display:flex;align-items:center;gap:16px;border:1px solid #cfc4c5;
  border-radius:0;padding:14px 16px;flex-shrink:0}
.cj-proc-row.done{opacity:.55}
.cj-proc-row.waiting{opacity:.45}
.cj-proc-thumb{width:48px;height:48px;flex-shrink:0;border-radius:0;object-fit:cover;
  filter:grayscale(1);opacity:.6}
.cj-proc-thumb-ph{width:48px;height:48px;flex-shrink:0;background:#e7e8e9}
.cj-proc-body{flex:1;min-width:0;display:flex;flex-direction:column;gap:10px}
.cj-proc-top{display:flex;justify-content:space-between;align-items:center;gap:12px}
.cj-proc-name{font-family:'Hanken Grotesk',sans-serif;font-size:14px;font-weight:600;
  color:#1b1b1b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cj-proc-status{font-family:'Hanken Grotesk',sans-serif;font-size:12px;color:#5d5e66;
  white-space:nowrap}
.cj-proc-row.active .cj-proc-status{animation:cjpulse 1.4s ease-in-out infinite}
@keyframes cjpulse{0%,100%{opacity:1}50%{opacity:.45}}
.cj-proc-track{width:100%;height:2px;background:#e1e3e4;position:relative;overflow:hidden}
.cj-proc-fill{position:absolute;top:0;left:0;height:100%;background:#1b1b1b;width:0}
.cj-proc-row.active .cj-proc-fill{animation:cjfill 8s cubic-bezier(.2,.7,.2,1) forwards}
.cj-proc-row.done .cj-proc-fill{width:100%}
.cj-proc-row.waiting .cj-proc-fill{width:6%}
@keyframes cjfill{0%{width:0}100%{width:92%}}"""


def _thumb_data_url(pil_img: Image.Image) -> str:
    thumb = pil_img.copy()
    thumb.thumbnail((160, 160))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _processing_card_html(names: list, thumbs: list, active: int) -> str:
    """Batch progress card shown in place of the drop zone.

    One row per uploaded image — done rows show a full bar, the active row an
    animated one, waiting rows a stub. The row list scrolls inside the card
    when there are more images than fit (the card itself keeps its size).
    """
    n = len(names)
    pct = round(active / n * 100)
    rows = []
    for i, (name, thumb) in enumerate(zip(names, thumbs)):
        state = "done" if i < active else ("active" if i == active else "waiting")
        status = {"done": "Done", "active": "Optimizing&hellip;",
                  "waiting": "Waiting&hellip;"}[state]
        img = (f"<img class='cj-proc-thumb' src='{thumb}'/>" if thumb
               else "<div class='cj-proc-thumb-ph'></div>")
        rows.append(
            f"<div class='cj-proc-row {state}'>{img}"
            "<div class='cj-proc-body'><div class='cj-proc-top'>"
            f"<span class='cj-proc-name'>{name}</span>"
            f"<span class='cj-proc-status'>{status}</span></div>"
            "<div class='cj-proc-track'><div class='cj-proc-fill'></div></div>"
            "</div></div>"
        )
    head = ""
    if n > 1:
        head = (
            "<div><div class='cj-proc-head'>"
            "<span class='cj-proc-head-label'>Batch Progress</span>"
            f"<span class='cj-proc-head-pct'>{pct}%</span></div>"
            "<div class='cj-proc-head-track'>"
            f"<div class='cj-proc-head-fill' style='width:{pct}%'></div></div></div>"
        )
    return (
        f"<div class='cj-proc'>{head}"
        f"<div class='cj-proc-list'>{''.join(rows)}</div>"
        "<a class='cj-proc-cancel' href='/' target='_self'>Cancel</a>"
        "</div>"
    )


def _pending_tiles_html(urls: list, active: int) -> str:
    """Filmstrip tiles for photos still processing (added via the + tile).

    Tiles before `active` are done (plain), `active` shows a spinner overlay,
    later ones wait dimmed. Rendered into a st.empty slot inside the strip.
    """
    tiles = []
    for k, u in enumerate(urls):
        cls = "" if k < active else (" spin" if k == active else " dim")
        style = f" style=\"background-image:url('{u}')\"" if u else ""
        tiles.append(f"<div class='cj-pend{cls}'{style}></div>")
    return "<div class='cj-pend-row'>" + "".join(tiles) + "</div>"


def _strip_html() -> str:
    return (
        "<div class='cj-strip'>"
        "<div class='cj-step'><span class='cj-num'>1</span>"
        "<span class='cj-steplbl'>Upload</span></div>"
        "<span class='cj-dash'>&mdash;</span>"
        "<div class='cj-step'><span class='cj-num'>2</span>"
        "<span class='cj-steplbl'>Algorithm Optimize</span></div>"
        "<span class='cj-dash'>&mdash;</span>"
        "<div class='cj-step'><span class='cj-num'>3</span>"
        "<span class='cj-steplbl'>Export</span></div>"
        "</div>"
    )


if _VIEW == "social":
    # Social Media Generator module (social.py) inside the same app shell.
    st.markdown(
        f"<style>{_SHELL_CSS}</style>" + _sidebar_html("social"),
        unsafe_allow_html=True,
    )
    import social

    social.render()
    st.stop()

# ── Listing view shell (fixed left sidebar) ────────────────────────────────────
st.markdown(
    f"<style>{_SHELL_CSS}</style>" + _sidebar_html("upload"),
    unsafe_allow_html=True,
)

# ── upload screen (homepage) ────────────────────────────────────────────────────
if not st.session_state.photos:
    st.markdown(
        f"<style>{_HOME_EXTRA_CSS}{_PROC_CSS}</style><div class='cj-home-top'></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.warn:
        st.markdown(
            f"<div class='warn-box'>&#9888; {st.session_state.warn}</div>",
            unsafe_allow_html=True,
        )

    # The drop zone lives in its own slot so it can be swapped, in place, for the
    # progress card while the photos process — no loading UI anywhere else.
    drop_slot = st.empty()
    with drop_slot.container():
        uploaded = st.file_uploader(
            "Drop photo here",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    st.markdown(_strip_html(), unsafe_allow_html=True)

    if uploaded:
        # Replace the drop zone with the batch progress card. Thumbnails for
        # every file render up front; the card re-renders in place before each
        # file so its row flips waiting → optimizing → done as the batch runs.
        _names, _thumbs = [], []
        for _f in uploaded:
            _names.append(getattr(_f, "name", "photo"))
            try:
                _f.seek(0)
                _thumbs.append(_thumb_data_url(ImageOps.exif_transpose(Image.open(_f))))
            except Exception:
                _thumbs.append(None)

        _warnings = []
        for _i, _f in enumerate(uploaded):
            drop_slot.markdown(
                _processing_card_html(_names, _thumbs, _i),
                unsafe_allow_html=True,
            )
            _w = _process_one(_f)
            if _w:
                _warnings.append(_w)
        st.session_state.warn = "  ".join(_warnings) if _warnings else None

        if not st.session_state.photos:
            st.rerun()            # everything failed — show the warning, stay here
        else:
            st.session_state.active = 0
            st.session_state.animate = True   # play before→after reveal once
            st.session_state._before_cache = {}   # invalidate cached before images
            _sync_active_mirror(st.session_state.photos[0])
            st.rerun()

# ── result screen ──────────────────────────────────────────────────────────────
else:
    # Keep the active index + mirror valid (e.g. after a reset elsewhere).
    st.session_state.active = min(st.session_state.active,
                                  len(st.session_state.photos) - 1)
    _photo = _active_photo()

    # ── Build the current before / after images for the active photo ──────────
    _t_rebuild = time.perf_counter()
    _base_result = _photo["result"]
    st.session_state["subject_rgba"] = _photo["subject_rgba"]
    st.session_state["subject_mask"] = _photo["subject_mask"]
    _caption = _photo["applied_description"] if _photo["text_mode"] else ""
    # Compose the full 1:1 square (which the user has confirmed never clips), draw the
    # caption on it, THEN pad out to the target ratio — so nothing is ever cropped.
    if _photo["text_mode"]:
        _t0 = time.perf_counter()
        _square = apply_text_overlay(_base_result, "1:1", _caption)
        log.info("PERF apply_text_overlay: %.0f ms", (time.perf_counter() - _t0) * 1000)
    else:
        _square = _base_result

    if _caption.strip():
        _square = _add_description_text(_square, _caption.strip())

    _after = _fit_to_ratio(_square, _photo["ratio"])

    _t0 = time.perf_counter()
    _png_bytes = _to_png_bytes(_after)
    _after_url = f"data:image/png;base64,{base64.b64encode(_png_bytes).decode()}"

    # before depends only on (active photo, ratio); cache its data URL per
    # (index, ratio) so a text/description edit doesn't re-crop and re-encode it.
    _bcache = st.session_state.setdefault("_before_cache", {})
    _bkey = (st.session_state.active, _photo["ratio"])
    _before_url = _bcache.get(_bkey)
    if _before_url is None:
        _before = _fit_to_ratio(_photo["original"], _photo["ratio"])
        _before_url = f"data:image/png;base64,{base64.b64encode(_to_png_bytes(_before)).decode()}"
        _bcache[_bkey] = _before_url
    log.info(
        "PERF after png+base64: %.0f ms (after=%.1f MB) | TOTAL rebuild %.0f ms",
        (time.perf_counter() - _t0) * 1000, len(_png_bytes) / 1e6,
        (time.perf_counter() - _t_rebuild) * 1000,
    )
    _dl_name = f"cj_listing_{st.session_state.active + 1}_{_photo['ratio'].replace(':', 'x')}.png"

    # Reveal animation plays exactly once, right after processing finishes.
    _animate = bool(st.session_state.pop("animate", False))

    # Canvas iframe needs an explicit height; size it to the aspect ratio.
    _AR = _RATIO_AR.get(_photo["ratio"], 1.0)                   # width / height
    _multi = len(st.session_state.photos) > 1
    # leave room under the split view for the filmstrip when there's >1 photo
    _canvas_h = max(520, min(820, round(767 / _AR))) - (120 if _multi else 0)
    st.markdown(f"<style>{_RESULT_CSS}{_MULTI_CSS}</style>", unsafe_allow_html=True)
    # Image preview on the LEFT, edit panel on the RIGHT.
    _canvas_col, _panel_col = st.columns([13, 7], gap="large")

    # ── Canvas: auto before→after reveal, then draggable compare slider ───────
    # Rendered in an iframe (components.html) because st.markdown strips <script>
    # and inline event handlers, which the drag/reveal logic needs.
    with _canvas_col, st.container(key="cj_canvas"):
        _anim_js = "true" if _animate else "false"
        _canvas_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{height:100%;background:#fff;font-family:'Hanken Grotesk',sans-serif}}
.cj-stage{{height:100vh;display:flex;align-items:center;justify-content:center;padding:48px}}
.cj-split{{position:relative;max-width:100%;line-height:0;border-radius:0;
  box-shadow:0 1px 3px rgba(0,0,0,.14),0 6px 24px rgba(0,0,0,.06);
  overflow:hidden;background:#f3f4f5;cursor:ew-resize;user-select:none;-webkit-user-select:none}}
/* the after image is in-flow and sizes the card; before + slider overlay it */
.cj-after-img{{position:relative;display:block;max-width:100%;max-height:calc(100vh - 96px);
  width:auto;height:auto;z-index:1;pointer-events:none}}
.cj-before-wrap{{position:absolute;inset:0;overflow:hidden;z-index:2;
  clip-path:polygon(0 0,50% 0,50% 100%,0 100%)}}
.cj-before-wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
  pointer-events:none}}
.cj-slider{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;z-index:5;
  box-shadow:0 0 0 1px rgba(0,0,0,.05),0 0 8px rgba(0,0,0,.3)}}
.cj-slider::after{{content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:34px;height:34px;background:#fff;border-radius:50%;border:2px solid #cfc4c5;
  box-shadow:0 2px 8px rgba(0,0,0,.25)}}
.cj-tag{{position:absolute;bottom:16px;z-index:6;padding:6px 14px;
  font-size:12px;font-weight:500;line-height:1;letter-spacing:.02em;cursor:pointer;
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);
  transition:opacity .15s ease,background .15s ease,color .15s ease}}
.cj-tag-l{{left:16px;background:rgba(248,249,250,.85);color:#1b1b1b}}
.cj-tag-r{{right:16px;background:rgba(27,27,27,.85);color:#fff}}
/* hover: Original mirrors "Try another photo", Enhanced mirrors "Export image" */
.cj-tag-l:hover{{background:#f8f9fa;color:#000}}
.cj-tag-r:hover{{opacity:.9}}
</style></head><body>
<div class="cj-stage">
<div class="cj-split" id="cjSplit">
  <img class="cj-after-img" src="{_after_url}"/>
  <div class="cj-before-wrap" id="cjBefore"><img src="{_before_url}"/></div>
  <div class="cj-slider" id="cjSlider"></div>
  <span class="cj-tag cj-tag-l" id="cjOrig">Original</span>
  <span class="cj-tag cj-tag-r" id="cjEnh">Enhanced</span>
</div>
</div>
<script>
(function(){{
  var sp=document.getElementById('cjSplit'),sl=document.getElementById('cjSlider'),
      bf=document.getElementById('cjBefore'),lo=document.getElementById('cjOrig'),
      le=document.getElementById('cjEnh');
  var raf=null,interacted=false,dragging=false,cur=0;
  function setX(x){{x=Math.max(0,Math.min(100,x));cur=x;sl.style.left=x+'%';
    bf.style.clipPath='polygon(0 0,'+x+'% 0,'+x+'% 100%,0 100%)';}}
  function animateTo(target){{interacted=true;if(raf)cancelAnimationFrame(raf);
    var from=cur,t0=null,dur=450;
    function tick(ts){{if(!t0)t0=ts;var p=Math.min(1,(ts-t0)/dur);
      setX(from+(target-from)*ease(p));
      if(p<1)raf=requestAnimationFrame(tick);}}
    raf=requestAnimationFrame(tick);}}
  function fromEvent(cx){{var r=sp.getBoundingClientRect();setX((cx-r.left)/r.width*100);}}
  function stop(){{interacted=true;if(raf)cancelAnimationFrame(raf);}}
  function down(e){{if(e.target===lo||e.target===le)return;dragging=true;stop();
    fromEvent((e.touches?e.touches[0]:e).clientX);}}
  function move(e){{if(dragging)fromEvent((e.touches?e.touches[0]:e).clientX);}}
  function up(){{dragging=false;}}
  sp.addEventListener('mousedown',down);
  window.addEventListener('mousemove',move);
  window.addEventListener('mouseup',up);
  sp.addEventListener('touchstart',down,{{passive:true}});
  window.addEventListener('touchmove',move,{{passive:true}});
  window.addEventListener('touchend',up);
  lo.addEventListener('click',function(e){{e.stopPropagation();animateTo(100);}});
  le.addEventListener('click',function(e){{e.stopPropagation();animateTo(0);}});
  function ease(t){{return t<0.5?2*t*t:1-Math.pow(-2*t+2,2)/2;}}
  if({_anim_js}){{
    var dur=2000,t0=null;setX(100);
    function step(ts){{if(interacted)return;if(!t0)t0=ts;
      var p=Math.min(1,(ts-t0)/dur);setX(100*(1-ease(p)));
      if(p<1)raf=requestAnimationFrame(step);}}
    raf=requestAnimationFrame(step);
  }}else{{setX(0);}}
}})();
</script></body></html>"""
        with st.container(key="cj_stage"):
            components.html(_canvas_doc, height=_canvas_h, scrolling=False)
            # White wash + frying-pan loader; CSS reveals it while Streamlit reruns.
            st.markdown(
                "<div class='cj-loading-overlay'><span class='loader'></span></div>",
                unsafe_allow_html=True,
            )

        # ── Filmstrip: click a thumbnail to switch the active photo. Each photo
        #    carries its own edit state, so switching never leaks edits across. ──
        if _multi:
            with st.container(key="cj_film"):
                _thumb_css = []
                for _i, _p in enumerate(st.session_state.photos):
                    _turl = _thumb_data_url(_p["result"])
                    # !important beats the app's global .stButton background shorthand
                    _thumb_css.append(
                        f".st-key-cj_thumb_{_i} button{{background-image:url('{_turl}') !important}}"
                    )
                    # wrapper anchors the hover ✕ chip to the tile's corner
                    with st.container(key=f"cj_tw_{_i}"):
                        st.button(" ", key=f"cj_thumb_{_i}",
                                  on_click=_activate, args=(_i,))
                        st.button("✕", key=f"cj_delbtn_{_i}",
                                  on_click=_ask_delete, args=(_i,))
                # slot where freshly added photos appear (with spinner) while
                # they process — instead of loading UI under the + tile
                _pend_slot = st.empty()
                # "+ Add Photos" tile — a compact uploader keyed by a nonce so a
                # completed add doesn't immediately re-fire on the next rerun.
                _more_key = f"cj_more_{st.session_state.get('more_nonce', 0)}"
                _more = st.file_uploader(
                    "Add photos", type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=True, key=_more_key,
                    label_visibility="collapsed",
                )
                _thumb_css.append(
                    f".st-key-cj_thumb_{st.session_state.active} button"
                    "{border:2px solid #1b1b1b !important;filter:none !important;"
                    "opacity:1 !important}"
                )
                st.markdown("<style>" + "".join(_thumb_css) + "</style>",
                            unsafe_allow_html=True)
            if _more:
                # show the new tiles in the strip immediately, then process one
                # by one — the spinner overlay walks across the pending tiles
                _urls = []
                for _f in _more:
                    try:
                        _f.seek(0)
                        _urls.append(_thumb_data_url(
                            ImageOps.exif_transpose(Image.open(_f))))
                    except Exception:
                        _urls.append(None)
                _warnings = []
                for _j, _f in enumerate(_more):
                    _pend_slot.markdown(_pending_tiles_html(_urls, _j),
                                        unsafe_allow_html=True)
                    _w = _process_one(_f)
                    if _w:
                        _warnings.append(_w)
                st.session_state.warn = "  ".join(_warnings) if _warnings else None
                st.session_state.more_nonce = st.session_state.get("more_nonce", 0) + 1
                st.rerun()

    # ── Right edit panel (nesting mirrors Figma: card → header / body / footer;
    #    body → group1 (Output Ratio) + group2 (Text Overlay)) ──────────────────
    with _panel_col:
        with st.container(border=True, key="cj_card"):
            st.markdown(
                "<div class='cj-panel-head'><h3>Adjust</h3></div>",
                unsafe_allow_html=True,
            )

            with st.container(key="cj_body"):
                # ── Group 1: Output Ratio (title + 3 cards) ──
                with st.container(key="cj_grp1"):
                    st.markdown("<p class='cj-sec'>Output Ratio</p>", unsafe_allow_html=True)
                    _ratio_cols = st.columns(3, gap="small")
                    for _i, _r in enumerate(("1:1", "4:3", "4:5")):
                        with _ratio_cols[_i]:
                            # on_click sets the ratio before the rerun, so the canvas
                            # recomputes once (no extra manual st.rerun()).
                            st.button(
                                _r,
                                key=f"cjr_{_r.replace(':', 'x')}",
                                type="primary" if st.session_state.ratio == _r else "secondary",
                                use_container_width=True,
                                on_click=_pick_ratio,
                                args=(_r,),
                            )

                # ── Group 2: Text Overlay (label+switch row, then input) ──
                with st.container(key="cj_grp2"):
                    # Real switch — only the switch is clickable (label is inert).
                    # on_change syncs text_mode before the rerun (single recompute).
                    # value comes from the cj_txt session key (kept in sync with
                    # the active photo); passing value= too triggers a Streamlit
                    # "default value but also set via Session State" warning.
                    st.toggle(
                        "Text Overlay",
                        key="cj_txt",
                        on_change=_sync_text_mode,
                    )

                    # Input is always rendered; the toggle reveals it purely in CSS.
                    with st.container(key="cj_desc"):
                        st.text_input(
                            "Description text",
                            placeholder="Description for bottom-right overlay...",
                            label_visibility="collapsed",
                            key="description_text",
                        )
                        # Small Apply button pinned to the right edge of the input
                        # row; CSS hides it while the field is empty. Clicking it
                        # commits the text as the caption drawn below the image.
                        st.button("Apply", key="cj_apply", on_click=_apply_caption)

            # ── Footer: Export + Try another (reset) ──
            # One photo → download it directly. Several → open the selection
            # modal (default: all selected) and export the chosen ones.
            with st.container(key="cj_foot"):
                if _multi:
                    st.button(
                        f"Export all ({len(st.session_state.photos)})",
                        key="cj_export_all", type="primary",
                        on_click=_open_export, use_container_width=True,
                    )
                else:
                    st.download_button(
                        "Export image", data=_png_bytes, file_name=_dl_name,
                        mime="image/png", key="cj_export_one",
                        use_container_width=True,
                    )
                st.button(
                    "Try another photo", key="cj_tryanother",
                    on_click=_reset_photos, use_container_width=True,
                )

    # ── Export selection modal (Figma 233:833) ─────────────────────────────────
    @st.dialog("Export selection", width="large")
    def _export_dialog():
        photos = st.session_state.photos
        sel = st.session_state.export_sel
        st.markdown(f"<style>{_EXPORT_CSS}</style>", unsafe_allow_html=True)
        _cols = st.columns(4)
        for _i, _p in enumerate(photos):
            with _cols[_i % 4], st.container(key=f"cj_ex_{_i}"):
                _is_sel = _i in sel
                # pure-CSS check badge (✓ drawn via ::after) so it never depends
                # on the Material Symbols icon font loading inside the dialog
                st.markdown(
                    f"<div class='cj-ex-tile{' sel' if _is_sel else ''}'>"
                    f"<img src='{_thumb_data_url(_p['result'])}'/>"
                    f"<span class='cj-ex-check{' on' if _is_sel else ''}'></span></div>",
                    unsafe_allow_html=True,
                )
                st.button(" ", key=f"cj_exbtn_{_i}", on_click=_toggle_export,
                          args=(_i,))

        _n = len(sel)
        if _n == 1:
            _idx = next(iter(sel))
            _pp = photos[_idx]
            _data = _to_png_bytes(_compose_photo(_pp))
            _fname = f"cj_listing_{_idx + 1}_{_pp['ratio'].replace(':', 'x')}.png"
            _mime = "image/png"
        else:
            _zbuf = io.BytesIO()
            with zipfile.ZipFile(_zbuf, "w", zipfile.ZIP_DEFLATED) as _z:
                for _idx in sorted(sel):
                    _pp = photos[_idx]
                    _z.writestr(
                        f"cj_listing_{_idx + 1}_{_pp['ratio'].replace(':', 'x')}.png",
                        _to_png_bytes(_compose_photo(_pp)),
                    )
            _data = _zbuf.getvalue()
            _fname = "cj_listing_export.zip"
            _mime = "application/zip"

        with st.container(key="cj_ex_actions"):
            _a1, _a2 = st.columns(2)
            with _a1:
                # A dialog only closes when st.rerun() runs *inside* the dialog
                # function, so handle the click here rather than via on_click.
                if st.button("Cancel", key="cj_ex_cancel",
                             use_container_width=True):
                    st.session_state.show_export = False
                    st.rerun()
            with _a2:
                # The browser download fires on click; the rerun then closes the
                # modal. disabled while nothing is selected.
                if st.download_button(
                    f"Export selected ({_n})", data=_data, file_name=_fname,
                    mime=_mime, disabled=(_n == 0),
                    key="cj_ex_go", type="primary", use_container_width=True,
                ):
                    st.session_state.show_export = False
                    st.rerun()

    if st.session_state.show_export:
        _export_dialog()

    # ── Delete confirmation (filmstrip ✕) ──────────────────────────────────────
    @st.dialog("Delete photo")
    def _delete_dialog():
        _i = st.session_state.confirm_delete
        _photos = st.session_state.photos
        if _i is None or not (0 <= _i < len(_photos)):
            st.session_state.confirm_delete = None
            st.rerun()
        st.markdown(
            f"Remove **{_photos[_i]['name']}** from this batch? "
            "Its edits will be lost."
        )
        _d1, _d2 = st.columns(2)
        with _d1:
            if st.button("Cancel", key="cj_del_cancel", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()
        with _d2:
            if st.button("Delete", key="cj_del_go", type="primary",
                         use_container_width=True):
                _delete_photo(_i)
                st.session_state.confirm_delete = None
                st.rerun()

    if st.session_state.confirm_delete is not None:
        _delete_dialog()
