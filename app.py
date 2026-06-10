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

# ── rembg model selection ──────────────────────────────────────────────────────
# u2net          - general-purpose, 173 MB, good baseline
# isnet-general-use - newer architecture, slightly higher quality on product shots
# Override via env: CJ_REMBG_MODEL=isnet-general-use
_REMBG_MODEL: str = os.environ.get("CJ_REMBG_MODEL", "u2net")

# ── page config ────────────────────────────────────────────────────────────────
# ?view=social shows the embedded IAAC tool; default is the Listing formatter.
_VIEW = st.query_params.get("view", "listing")

st.set_page_config(
    page_title="CJ Listing Formatter",
    layout="wide" if _VIEW == "social" else "centered",
    initial_sidebar_state="collapsed",
)

PRIMARY = "#C4D938"
DARK = "#555555"

st.markdown(
    f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: #f8f8f5; }}
  .block-container {{ max-width: 880px; padding-top: 2.5rem; padding-bottom: 3rem; }}
  header[data-testid="stHeader"] {{ background: transparent; box-shadow: none; }}
  [data-testid="stFileUploader"] {{
    border: 2px dashed {PRIMARY};
    border-radius: 14px;
    padding: 2.5rem 1rem;
    background: #fff;
    text-align: center;
  }}
  .stDownloadButton > button,
  .stButton > button {{
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.45rem 1.4rem !important;
    transition: background 0.15s;
    white-space: nowrap !important;
  }}
  /* Secondary / default buttons */
  .stButton > button {{
    background: #e0e0e0 !important;
    color: #555 !important;
  }}
  .stButton > button:hover {{
    background: #cacaca !important;
    color: #555 !important;
  }}
  /* Primary buttons (active ratio chip) + download button */
  .stDownloadButton > button,
  .stButton > button[kind="primary"] {{
    background: {PRIMARY} !important;
    color: #006633 !important;
  }}
  .stDownloadButton > button:hover,
  .stButton > button[kind="primary"]:hover {{
    background: #afc227 !important;
    color: #006633 !important;
  }}
  /* Ratio chip row: tighter padding, smaller font */
  .ratio-row .stButton > button {{
    padding: 0.3rem 0 !important;
    font-size: 0.85rem !important;
  }}
  /* Text inputs: dark-green focus ring instead of theme red */
  [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{
    border-color: #006633 !important;
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
    background:{PRIMARY}; color:#006633;
    border-radius:8px; padding:0.5rem 1.4rem;
    font-weight:700; font-size:0.9rem;
    text-decoration:none; white-space:nowrap;
  }}
  .cj-lb-dl:hover {{ background:#afc227; color:#006633; }}
  .col-label {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 6px;
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

# ── rembg session (loaded once, cached for the lifetime of the server) ─────────

@st.cache_resource(show_spinner=False)
def _rembg_session():
    from rembg import new_session

    log.info("Loading rembg model '%s'...", _REMBG_MODEL)
    t0 = time.perf_counter()
    try:
        session = new_session(_REMBG_MODEL)
    except Exception as exc:
        log.error(
            "Failed to initialise rembg model '%s': %s",
            _REMBG_MODEL, exc, exc_info=True,
        )
        raise RuntimeError(
            f'Could not load background-removal model "{_REMBG_MODEL}". '
            "Check your internet connection. The first run downloads the model. "
            f"Detail: {exc}"
        ) from exc

    log.info("rembg session ready in %.1f s", time.perf_counter() - t0)
    return session


# ── image processing ───────────────────────────────────────────────────────────

def _extract_subject(pil_img: Image.Image):
    """
    Remove background with rembg/U2Net and return (rgb, alpha) as uint8 arrays.
    alpha is a smooth 0-255 channel — no hard thresholding done here.
    """
    from rembg import remove as rembg_remove

    log.info(
        "Removing background from %dx%d image using model '%s'...",
        pil_img.width, pil_img.height, _REMBG_MODEL,
    )
    t0 = time.perf_counter()

    try:
        rgba = rembg_remove(pil_img.convert("RGBA"), session=_rembg_session())
    except RuntimeError:
        raise
    except MemoryError as exc:
        log.error("OOM during rembg inference: %s", exc, exc_info=True)
        raise RuntimeError(
            "Not enough memory to process this image. "
            "Try a smaller image (< 4000 x 4000 px)."
        ) from exc
    except Exception as exc:
        log.error("rembg.remove() failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"Background removal failed ({type(exc).__name__}). "
            "Try a different image or restart the app."
        ) from exc

    log.info("Background removed in %.2f s", time.perf_counter() - t0)
    arr = np.array(rgba)
    return arr[:, :, :3], arr[:, :, 3]


def _tighten_alpha(alpha: np.ndarray) -> np.ndarray:
    """
    Post-process rembg alpha:
    1. Remove isolated noise specks (small open/close)
    2. Fill topologically enclosed holes (e.g. ring centre, frame window)
    3. Hard-zero pixels below confidence threshold
       – rembg assigns alpha 0-80 to background gaps between limbs;
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

    This catches concave "gap" areas (e.g. between arm and body) that rembg
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
    sh_y = max(1, int(bh_s * 0.06))

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
    Full pipeline: rembg subject extraction → crop → 1600×1600 canvas + shadow.
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
    ch_c, cw_c = crop_rgb.shape[:2]
    nw = max(1, int(cw_c * scale))
    nh = max(1, int(ch_c * scale))
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


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
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


def _crop_result(base: Image.Image, ratio: str) -> Image.Image:
    """Crop the 1600×1600 base canvas to the target aspect ratio (center crop)."""
    W, H = base.size
    ar = _RATIO_AR.get(ratio, 1.0)
    if ar >= 1.0:
        new_h = int(W / ar)
        top = (H - new_h) // 2
        return base.crop((0, top, W, top + new_h))
    else:
        new_w = int(H * ar)
        left = (W - new_w) // 2
        return base.crop((left, 0, left + new_w, H))


def _crop_before(pil_img: Image.Image, ratio: str) -> Image.Image:
    """Center-crop the original photo to match the target aspect ratio."""
    w, h = pil_img.size
    ar = _RATIO_AR.get(ratio, 1.0)
    if w / h >= ar:
        new_w = int(h * ar)
        x0 = (w - new_w) // 2
        return pil_img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = int(w / ar)
        y0 = (h - new_h) // 2
        return pil_img.crop((0, y0, w, y0 + new_h))


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

# ── Nav pill + view router ─────────────────────────────────────────────────────
# Both tools run on this single Streamlit server, so the app works when
# deployed (e.g. Streamlit Cloud) where only one port is exposed.
# ?view=social renders the IAAC app inline via st.components.v1.html.
_NAV_CSS = """\
#cj-nav{display:inline-flex;gap:4px;background:#fff;border-radius:8px;padding:4px;
        box-shadow:0 2px 10px rgba(0,0,0,.10)}
#cj-nav .cj-np{padding:5px 16px;border-radius:6px;border:none;
  font-size:.83rem;font-weight:600;cursor:default;background:none;
  color:#888 !important;text-decoration:none !important;
  display:inline-block;transition:background .15s,color .15s;
  font-family:inherit;line-height:1.4}
#cj-nav .cj-np.active{background:#FDECEA;color:#E8605A !important}
#cj-nav a.cj-np{cursor:pointer;color:#888 !important;text-decoration:none !important}
#cj-nav .cj-np:not(.active):hover{background:#f5f5f5}"""

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
    return f'<div id="cj-nav">{listing}{social}</div>'


if _VIEW == "social":
    st.markdown(
        f"<style>{_NAV_CSS}\n"
        ".block-container{max-width:100% !important;"
        "padding-top:1.2rem !important;padding-bottom:0.5rem !important}</style>"
        + _nav_pill("social"),
        unsafe_allow_html=True,
    )
    components.html(_iaac_component_html(), height=860, scrolling=True)
    st.stop()

st.markdown(
    f"<style>{_NAV_CSS}</style>"
    + _nav_pill("listing")
    + f"<h2 style='color:{DARK};margin:0 0 1.4rem;font-size:1.85rem;font-weight:700;'>"
    "CJ Listing Formatter</h2>",
    unsafe_allow_html=True,
)

# ── upload screen ──────────────────────────────────────────────────────────────
if st.session_state.result is None:
    uploaded = st.file_uploader(
        "Drop photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<p style='text-align:center;color:#bbb;font-size:0.78rem;margin-top:-0.3rem;'>"
        "JPG · PNG · WEBP · up to 200 MB</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.warn:
        st.markdown(
            f"<div class='warn-box'>&#9888; {st.session_state.warn}</div>",
            unsafe_allow_html=True,
        )

    if uploaded is not None:
        pil_img = ImageOps.exif_transpose(Image.open(uploaded))
        with st.spinner("Processing…"):
            result, warn = make_listing(pil_img)

        if warn:
            st.session_state.warn = warn
            st.rerun()
        else:
            st.session_state.original = pil_img
            st.session_state.result = result
            st.session_state.warn = None
            st.rerun()

# ── result screen ──────────────────────────────────────────────────────────────
else:
    # ── Ratio selector ──────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#aaa;margin-bottom:0.3rem;'>Output ratio</p>",
        unsafe_allow_html=True,
    )
    _RATIO_OPTS = [("1:1", "eBay"), ("4:3", "Website"), ("4:5", "Instagram")]
    _ratio_cols = st.columns(3)
    for _i, (_r, _hint) in enumerate(_RATIO_OPTS):
        with _ratio_cols[_i]:
            if st.button(
                _r,
                key=f"_ratio_{_r}",
                type="primary" if st.session_state.ratio == _r else "secondary",
                use_container_width=True,
            ):
                st.session_state.ratio = _r
                st.rerun()
            st.markdown(
                f"<p style='text-align:center;color:#bbb;font-size:0.72rem;"
                f"margin-top:-0.5rem;'>{_hint}</p>",
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Text overlay toggle ───────────────────────────────────────────────────
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#aaa;margin-bottom:0.5rem;'>Text format</p>",
        unsafe_allow_html=True,
    )

    _tm_cols = st.columns([1, 4])
    with _tm_cols[0]:
        _toggle_label = "On ✓" if st.session_state.text_mode else "Off"
        if st.button(
            _toggle_label,
            key="_toggle_text",
            type="primary" if st.session_state.text_mode else "secondary",
            use_container_width=True,
        ):
            st.session_state.text_mode = not st.session_state.text_mode
            st.rerun()
    with _tm_cols[1]:
        st.markdown(
            "<p style='color:#999;font-size:0.82rem;margin:0.4rem 0 0;'>"
            "Add bold text behind your product — great for social listings.</p>",
            unsafe_allow_html=True,
        )

    # ── Description text input (only in text mode) ────────────────────────────
    if st.session_state.text_mode:
        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#aaa;margin-bottom:0.5rem;'>Description</p>",
            unsafe_allow_html=True,
        )
        _desc_col, _apply_col, _clear_col = st.columns([4, 1, 1], vertical_alignment="bottom")
        with _desc_col:
            st.text_input(
                "Description text",
                placeholder="e.g. Premium condition",
                label_visibility="collapsed",
                key="description_text",
            )
        with _apply_col:
            # Clicking blurs the input, which commits its value before the rerun.
            st.button("Apply", type="primary", use_container_width=True)
        with _clear_col:
            # Callback runs before the rerun, so the emptied value wins over
            # whatever the blur just committed.
            st.button(
                "Clear",
                use_container_width=True,
                on_click=lambda: st.session_state.update(description_text=""),
            )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    _base_result = st.session_state.result

    # Only apply the background/text overlay when text mode is enabled.
    if st.session_state.text_mode:
        _display_result = apply_text_overlay(_base_result, st.session_state.ratio, st.session_state.description_text)
    else:
        _display_result = _base_result

    _before = _crop_before(st.session_state.original, st.session_state.ratio)
    _after  = _crop_result(_display_result, st.session_state.ratio)
    
    # Add description text if provided (text mode only — the input is hidden otherwise)
    if st.session_state.text_mode and st.session_state.description_text.strip():
        _after = _add_description_text(_after, st.session_state.description_text.strip())

    _png_bytes = _to_png_bytes(_after)
    _b64 = base64.b64encode(_png_bytes).decode()
    _data_url = f"data:image/png;base64,{_b64}"

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='col-label'>Before</div>", unsafe_allow_html=True)
        st.image(_before, use_container_width=True)
    with col2:
        _dl_name = f"cj_listing_{st.session_state.ratio.replace(':','x')}.png"
        st.markdown("<div class='col-label'>After</div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="cj-after" id="cj-wrap">
              <img id="cj-thumb" src="{_data_url}"
                   style="width:100%;border-radius:8px;display:block;cursor:zoom-in;" />
              <button id="cj-expand" class="cj-expand-btn">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="#555555">
                  <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                </svg>
              </button>
            </div>
            <div id="cj-lb">
              <img id="cj-lb-img" src="{_data_url}" />
              <a class="cj-lb-dl" id="cj-lb-dl" href="{_data_url}"
                 download="{_dl_name}">↓ Download</a>
            </div>
            <img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
                 style="display:none"
                 onload="(function(){{
                   var t=document.getElementById('cj-thumb'),
                       b=document.getElementById('cj-expand'),
                       lb=document.getElementById('cj-lb'),
                       im=document.getElementById('cj-lb-img'),
                       dl=document.getElementById('cj-lb-dl');
                   function op(){{lb.style.display='flex';}}
                   function cl(){{lb.style.display='none';}}
                   if(t)t.addEventListener('click',op);
                   if(b)b.addEventListener('click',function(e){{e.stopPropagation();op();}});
                   if(lb)lb.addEventListener('click',function(e){{if(e.target===lb)cl();}});
                   if(im)im.addEventListener('click',function(e){{e.stopPropagation();}});
                   if(dl)dl.addEventListener('click',function(e){{e.stopPropagation();}});
                 }})();" />""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
    _, btn_dl, btn_try, _ = st.columns([2.5, 1.2, 1.5, 2.5])
    with btn_dl:
        st.download_button(
            "↓  Download",
            data=_png_bytes,
            file_name=f"cj_listing_{st.session_state.ratio.replace(':','x')}.png",
            mime="image/png",
        )
    with btn_try:
        if st.button("↺  Try another photo"):
            st.session_state.original = None
            st.session_state.result = None
            st.session_state.warn = None
            st.session_state.text_mode = False
            st.session_state.description_text = ""
            st.session_state.subject_mask = None
            st.session_state.subject_rgba = None
            st.rerun()
