import base64
import functools
import io
import json
import logging
import math
import os
import textwrap
import time

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
for _k in ("original", "result", "warn", "subject_mask", "subject_rgba"):
    if _k not in st.session_state:
        st.session_state[_k] = None
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


# ── Widget callbacks — mutate state *before* the rerun so the canvas at the top
#    of the result screen recomputes exactly once (no wasted extra rerun). ───────
def _pick_ratio(r):
    st.session_state.ratio = r

def _sync_text_mode():
    st.session_state.text_mode = st.session_state.cj_txt

def _apply_caption():
    st.session_state.applied_description = st.session_state.description_text

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
  min-height:0;margin-left:256px}
/* panel scrolls internally if its own content ever exceeds the viewport */
@PANEL@ [data-testid="stVerticalBlockBorderWrapper"]{overflow-y:auto}
@CANVAS@{flex:1 1 auto !important;width:auto !important;min-width:0}
@PANEL@{flex:0 0 384px !important;width:384px !important;min-width:384px !important}
/* canvas column stretches to the row (align-items:stretch); the inner chain
   fills that height. NB: don't set height on the column itself — a resolved
   percentage height there cancels the stretch and collapses it. */
@CANVAS@ [data-testid="stVerticalBlockBorderWrapper"],
@CANVAS@ [data-testid="stVerticalBlock"],
@CANVAS@ [data-testid="stElementContainer"]{height:100% !important;min-height:0}
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
.st-key-cj_canvas{position:relative}
/* Streamlit makes every stElementContainer position:relative, so the overlay would
   anchor to its own (below-the-iframe) container. Pull that container out of flow and
   pin it over cj_canvas so the overlay lands exactly on the image. */
.st-key-cj_canvas>[data-testid="stElementContainer"]:has(.cj-loading-overlay){
  position:absolute !important;inset:0 !important;margin:0 !important;height:auto !important;
  z-index:30;pointer-events:none !important}
/* only capture pointer events while a rerun is running, so the idle slider/buttons work */
[data-test-script-state="running"] .st-key-cj_canvas>[data-testid="stElementContainer"]:has(.cj-loading-overlay){
  pointer-events:auto !important}
.cj-loading-overlay{position:absolute;inset:0;z-index:30;display:flex;
  align-items:center;justify-content:center;background:rgba(255,255,255,.2);
  opacity:0;visibility:hidden;pointer-events:none;transition:opacity .12s ease}
[data-testid="stApp"][data-test-script-state="running"] .cj-loading-overlay,
[data-testid="stApp"][data-test-script-state="rerunRequested"] .cj-loading-overlay{
  opacity:1 !important;visibility:visible !important;pointer-events:auto}
/* keep the canvas + overlay containers un-dimmed while Streamlit marks them stale */
[data-test-script-state="running"] .st-key-cj_canvas>[data-testid="stElementContainer"]{
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


# ── In-place processing card (replaces the drop zone while a photo processes) ──
_PROC_CSS = """\
.cj-proc{position:relative;border:1px solid #cfc4c5;border-radius:12px;background:#fff;
  min-height:380px;display:flex;flex-direction:column;justify-content:center;gap:16px;padding:1.5rem}
.cj-proc-cancel,.cj-proc-cancel:link,.cj-proc-cancel:visited{position:absolute;left:0;right:0;
  bottom:20px;text-align:center;font-family:'Hanken Grotesk',sans-serif;font-size:12px;
  color:#9a9a9a !important;text-decoration:underline !important;text-underline-offset:2px}
.cj-proc-cancel:hover{color:#5d5e66 !important}
.cj-proc-row{display:flex;align-items:center;gap:16px;border:1px solid #cfc4c5;
  border-radius:8px;padding:16px}
.cj-proc-thumb{width:64px;height:64px;flex-shrink:0;border-radius:8px;object-fit:cover;
  filter:grayscale(1);opacity:.6}
.cj-proc-body{flex:1;min-width:0;display:flex;flex-direction:column;gap:10px}
.cj-proc-top{display:flex;justify-content:space-between;align-items:center;gap:12px}
.cj-proc-name{font-family:'Hanken Grotesk',sans-serif;font-size:14px;font-weight:600;
  color:#1b1b1b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cj-proc-status{font-family:'Hanken Grotesk',sans-serif;font-size:12px;color:#5d5e66;
  white-space:nowrap;animation:cjpulse 1.4s ease-in-out infinite}
@keyframes cjpulse{0%,100%{opacity:1}50%{opacity:.45}}
.cj-proc-track{width:100%;height:2px;background:#e1e3e4;position:relative;overflow:hidden}
.cj-proc-fill{position:absolute;top:0;left:0;height:100%;background:#1b1b1b;width:0;
  animation:cjfill 8s cubic-bezier(.2,.7,.2,1) forwards}
@keyframes cjfill{0%{width:0}100%{width:92%}}"""


def _thumb_data_url(pil_img: Image.Image) -> str:
    thumb = pil_img.copy()
    thumb.thumbnail((160, 160))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _processing_card_html(name: str, thumb_url: str) -> str:
    """Single-file progress card shown in place of the drop zone.

    (With one photo there is no batch-progress row — just this file's bar.)
    """
    return (
        "<div class='cj-proc'><div class='cj-proc-row'>"
        f"<img class='cj-proc-thumb' src='{thumb_url}'/>"
        "<div class='cj-proc-body'><div class='cj-proc-top'>"
        f"<span class='cj-proc-name'>{name}</span>"
        "<span class='cj-proc-status'>Optimizing&hellip;</span></div>"
        "<div class='cj-proc-track'><div class='cj-proc-fill'></div></div>"
        "</div></div>"
        "<a class='cj-proc-cancel' href='/' target='_self'>Cancel</a>"
        "</div>"
    )


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
if st.session_state.result is None:
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
    # progress card while the photo processes — no loading UI anywhere else.
    drop_slot = st.empty()
    with drop_slot.container():
        uploaded = st.file_uploader(
            "Drop photo here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

    st.markdown(_strip_html(), unsafe_allow_html=True)

    if uploaded is not None:
        pil_img = ImageOps.exif_transpose(Image.open(uploaded))

        # Replace the drop zone with the in-place progress bar, then process.
        drop_slot.markdown(
            _processing_card_html(uploaded.name, _thumb_data_url(pil_img)),
            unsafe_allow_html=True,
        )
        result, warn = make_listing(pil_img)

        if warn:
            st.session_state.warn = warn
            st.rerun()
        else:
            st.session_state.original = pil_img
            st.session_state.result = result
            st.session_state.warn = None
            st.session_state.animate = True   # play before→after reveal once
            st.session_state._before_cache = {}   # invalidate cached before images
            st.rerun()

# ── result screen ──────────────────────────────────────────────────────────────
else:
    # ── Build the current before / after images from session state ────────────
    _t_rebuild = time.perf_counter()
    _base_result = st.session_state.result
    _caption = st.session_state.applied_description if st.session_state.text_mode else ""
    # Compose the full 1:1 square (which the user has confirmed never clips), draw the
    # caption on it, THEN pad out to the target ratio — so nothing is ever cropped.
    if st.session_state.text_mode:
        _t0 = time.perf_counter()
        _square = apply_text_overlay(_base_result, "1:1", _caption)
        log.info("PERF apply_text_overlay: %.0f ms", (time.perf_counter() - _t0) * 1000)
    else:
        _square = _base_result

    if _caption.strip():
        _square = _add_description_text(_square, _caption.strip())

    _after = _fit_to_ratio(_square, st.session_state.ratio)

    _t0 = time.perf_counter()
    _png_bytes = _to_png_bytes(_after)
    _after_url = f"data:image/png;base64,{base64.b64encode(_png_bytes).decode()}"

    # before depends only on (original, ratio); cache its data URL per ratio so a
    # text/description edit doesn't re-crop and re-encode it. Cleared on new upload.
    _bcache = st.session_state.setdefault("_before_cache", {})
    _before_url = _bcache.get(st.session_state.ratio)
    if _before_url is None:
        _before = _fit_to_ratio(st.session_state.original, st.session_state.ratio)
        _before_url = f"data:image/png;base64,{base64.b64encode(_to_png_bytes(_before)).decode()}"
        _bcache[st.session_state.ratio] = _before_url
    log.info(
        "PERF after png+base64: %.0f ms (after=%.1f MB) | TOTAL rebuild %.0f ms",
        (time.perf_counter() - _t0) * 1000, len(_png_bytes) / 1e6,
        (time.perf_counter() - _t_rebuild) * 1000,
    )
    _dl_name = f"cj_listing_{st.session_state.ratio.replace(':', 'x')}.png"

    # Reveal animation plays exactly once, right after processing finishes.
    _animate = bool(st.session_state.pop("animate", False))

    # Canvas iframe needs an explicit height; size it to the aspect ratio.
    _AR = _RATIO_AR.get(st.session_state.ratio, 1.0)            # width / height
    _canvas_h = max(520, min(820, round(767 / _AR)))
    st.markdown(f"<style>{_RESULT_CSS}</style>", unsafe_allow_html=True)
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
        components.html(_canvas_doc, height=_canvas_h, scrolling=False)
        # White wash + frying-pan loader; CSS reveals it while Streamlit reruns.
        st.markdown(
            "<div class='cj-loading-overlay'><span class='loader'></span></div>",
            unsafe_allow_html=True,
        )

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
                    st.toggle(
                        "Text Overlay",
                        value=st.session_state.text_mode,
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

            # ── Footer: Export (data-URL download) + Try another (reset) ──
            st.markdown(
                "<div class='cj-foot'>"
                f"<a class='cj-btn-primary' href='{_after_url}' download='{_dl_name}'>Export image</a>"
                "<a class='cj-btn-outline' href='/' target='_self'>Try another photo</a>"
                "</div>",
                unsafe_allow_html=True,
            )
