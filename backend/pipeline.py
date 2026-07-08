"""CJ Studio listing pipeline — extracted from app.py, framework-free.

Every function that used to read/write st.session_state now takes and returns
explicit values, so the same code runs under FastAPI (or any host):

    make_listing(img)                    -> (result, warn, subject_mask, subject_rgba)
    compose(result, subject_rgba, ...)   -> final PIL image at the chosen ratio

Model handles are cached per-process with functools.lru_cache (the FastAPI app
runs one process; workers each warm their own copy, same as st.cache_resource).
"""
import functools
import io
import logging
import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

log = logging.getLogger("cj_pipeline")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Subjects whose longest side is below this get an AI upscale instead of a
# blurry LANCZOS enlarge before they're blown up to fill the canvas.
_UPSCALE_BELOW = int(os.environ.get("CJ_UPSCALE_BELOW", "900"))
_ESRGAN_ENABLED = os.environ.get("CJ_UPSCALE", "1") != "0"

_RATIO_AR = {"1:1": 1.0, "4:3": 4 / 3, "4:5": 4 / 5}


# ── models (cached per process) ────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
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


@functools.lru_cache(maxsize=1)
def _esrgan_model():
    """Load Real-ESRGAN x4 (via spandrel). Returns (net, device, scale) or None."""
    if not _ESRGAN_ENABLED:
        return None
    try:
        import torch
        from spandrel import ModelLoader
    except Exception as exc:
        log.info("Real-ESRGAN unavailable (%s) — skipping AI upscale.", exc)
        return None

    weight = os.path.join(_BASE_DIR, "models", "RealESRGAN_x4plus.pth")
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
    """Upscale HxWx3 uint8 with Real-ESRGAN; falls back to (rgb, 1)."""
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
            "Real-ESRGAN upscaled %dx%d -> %dx%d in %.2f s",
            rgb.shape[1], rgb.shape[0], out.shape[1], out.shape[0],
            time.perf_counter() - t0,
        )
        return out, factor
    except Exception as exc:
        log.warning("Real-ESRGAN inference failed (%s) — using LANCZOS.", exc)
        return rgb, 1


@functools.lru_cache(maxsize=1)
def load_bg_image():
    """Project background artwork (bg_artwork.png at repo root), or None."""
    path = os.path.join(_BASE_DIR, "bg_artwork.png")
    if not os.path.exists(path):
        return None
    try:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    except Exception:
        return Image.open(path).convert("RGBA")


# ── subject extraction ─────────────────────────────────────────────────────────

def _extract_subject(pil_img: Image.Image):
    """Remove background; returns (rgb, alpha) uint8 arrays (soft 0-255 matte)."""
    log.info("Removing background from %dx%d image...", pil_img.width, pil_img.height)
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
    """Speck cleanup, enclosed-hole fill, hard-zero low-confidence pixels."""
    _, binary = cv2.threshold(alpha, 15, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)

    inv = cv2.bitwise_not(binary)
    _, labels = cv2.connectedComponents(inv, connectivity=8)
    border_labels = set(np.concatenate([
        labels[0, :], labels[-1, :],
        labels[1:-1, 0], labels[1:-1, -1],
    ]).tolist())
    filled = binary.copy()
    for lbl in np.unique(labels):
        if lbl != 0 and lbl not in border_labels:
            filled[labels == lbl] = 255

    out = np.where(
        filled > 0,
        np.where(alpha > 0, alpha, np.uint8(255)),
        np.uint8(0),
    ).astype(np.uint8)
    out[out < 80] = 0
    return out


def _color_guided_cleanup(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Zero moderate-alpha pixels whose colour matches the background colour."""
    h, w = rgb.shape[:2]
    b = max(10, min(30, h // 20, w // 20))
    strips = [rgb[:b, :], rgb[-b:, :], rgb[:, :b], rgb[:, -b:]]
    border_px = np.concatenate([s.reshape(-1, 3) for s in strips]).astype(np.float32)
    if len(border_px) == 0:
        return alpha
    bg_color = np.median(border_px, axis=0)
    if float(np.mean(bg_color)) > 230:
        return alpha
    diff = rgb.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    out = alpha.copy()
    uncertain = (alpha >= 80) & (alpha <= 210)
    out[uncertain & (dist < 45)] = 0
    return out


def generate_product_shadow(r_mask, bx_s, by_s, bw_s, bh_s, ox, oy, target_cx, CS):
    """Simple soft drop shadow: blurred silhouette shifted slightly down-right."""
    H, W = r_mask.shape[:2]
    empty = np.zeros((CS, CS), dtype=np.float32)

    if int(np.count_nonzero(r_mask > 128)) < 100 or bw_s < 4 or bh_s < 4:
        return empty, empty

    sh_x = max(1, int(bw_s * 0.03))
    sh_y = max(1, int(bh_s * 0.02))

    shadow = empty.copy()
    s_oy, s_ox = oy + sh_y, ox + sh_x
    cy0 = max(0, s_oy); cy1 = min(CS, s_oy + H)
    cx0 = max(0, s_ox); cx1 = min(CS, s_ox + W)
    if cy1 > cy0 and cx1 > cx0:
        iy0 = cy0 - s_oy; iy1 = iy0 + (cy1 - cy0)
        ix0 = cx0 - s_ox; ix1 = ix0 + (cx1 - cx0)
        shadow[cy0:cy1, cx0:cx1] = r_mask[iy0:iy1, ix0:ix1].astype(np.float32) / 255.0

    shadow = cv2.GaussianBlur(shadow, (0, 0), 24.0)
    shadow = np.clip(shadow * 0.16, 0.0, 1.0)
    return shadow, empty


def make_listing(pil_img: Image.Image):
    """Full pipeline: subject extraction → crop → 1600x1600 canvas + shadow.

    Returns (result_pil, warn, subject_mask, subject_rgba); on failure the
    first element is None and warn holds the message.
    """
    log.info("make_listing: start input=%dx%d", pil_img.width, pil_img.height)
    t_total = time.perf_counter()

    try:
        rgb, alpha = _extract_subject(pil_img)
    except RuntimeError as exc:
        log.warning("make_listing: subject extraction failed - %s", exc)
        return None, str(exc), None, None

    alpha = _tighten_alpha(alpha)
    alpha = _color_guided_cleanup(rgb, alpha)

    h, w = rgb.shape[:2]
    total = h * w

    fg_px = int(np.count_nonzero(alpha > 15))
    if fg_px < total * 0.01:
        return None, "Could not detect the item. Try a photo with a cleaner background.", None, None

    pts = cv2.findNonZero((alpha > 15).astype(np.uint8) * 255)
    if pts is None:
        return None, "Could not detect the item. Try a photo with a cleaner background.", None, None
    bx, by, bw, bh = cv2.boundingRect(pts)

    pad = max(10, int(max(bw, bh) * 0.04))
    x1, y1 = max(0, bx - pad), max(0, by - pad)
    x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
    crop_rgb = rgb[y1:y2, x1:x2]
    crop_alpha = alpha[y1:y2, x1:x2]

    CS = 1600
    canvas = np.full((CS, CS, 3), [253, 253, 242], dtype=np.uint8)  # #FDFDF2

    scale = (CS * 0.75) / max(bw, bh)

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
    r_rgb = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    r_mask = cv2.resize(crop_alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    scx = int((bx - x1 + bw / 2) * scale)
    scy = int((by - y1 + bh / 2) * scale)

    target_cx = CS // 2
    target_cy = int(CS * 0.52)
    ox = target_cx - scx
    oy = target_cy - scy

    def _slices(offset: int, length: int, limit: int):
        c0 = max(0, offset)
        c1 = min(limit, offset + length)
        return slice(c0, c1), slice(c0 - offset, c1 - offset)

    can_y, img_y = _slices(oy, nh, CS)
    can_x, img_x = _slices(ox, nw, CS)

    bx_s = int((bx - x1) * scale)
    by_s = int((by - y1) * scale)
    bw_s = int(bw * scale)
    bh_s = int(bh * scale)

    cast_shadow, contact_shadow = generate_product_shadow(
        r_mask, bx_s, by_s, bw_s, bh_s, ox, oy, target_cx, CS,
    )

    total_shadow = np.clip(cast_shadow + contact_shadow, 0.0, 1.0)
    canvas_f = canvas.astype(np.float32)
    for c in range(3):
        canvas_f[:, :, c] = np.clip(canvas_f[:, :, c] * (1.0 - total_shadow), 0, 255)
    canvas = canvas_f.astype(np.uint8)

    alpha_f = r_mask[img_y, img_x].astype(np.float32) / 255.0
    roi = canvas[can_y, can_x].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = r_rgb[img_y, img_x, c] * alpha_f + roi[:, :, c] * (1.0 - alpha_f)
    canvas[can_y, can_x] = roi.astype(np.uint8)

    full_mask = np.zeros((CS, CS), dtype=np.uint8)
    full_mask[can_y, can_x] = r_mask[img_y, img_x]

    full_rgba = np.zeros((CS, CS, 4), dtype=np.uint8)
    full_rgba[can_y, can_x, :3] = r_rgb[img_y, img_x]
    full_rgba[can_y, can_x, 3] = r_mask[img_y, img_x]

    out = Image.fromarray(canvas)
    out = ImageEnhance.Brightness(out).enhance(1.02)
    out = ImageEnhance.Contrast(out).enhance(1.05)
    log.info(
        "make_listing: done in %.2f s output=%dx%d",
        time.perf_counter() - t_total, out.width, out.height,
    )
    return out, None, full_mask, full_rgba


# ── text overlay + composition ─────────────────────────────────────────────────

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
    }
    for path in candidates.get(style, []):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _split_words(words: list, max_lines: int) -> list:
    if not words:
        return [""]
    n = min(len(words), max_lines)
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def _render_background_text(canvas: Image.Image, CS: int) -> Image.Image:
    """Two-line Construction Junction text layer (fallback when no artwork)."""
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
    y = int(CS * 0.08)
    for line in lines:
        width = draw.textlength(line, font=font)
        x = (CS - width) // 2
        draw.text((x, y), line, font=font, fill=(210, 235, 95, 180))
        y += line_height

    out = canvas.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def apply_text_overlay(base: Image.Image, subject_rgba: np.ndarray,
                       bg_image: Image.Image, target_ratio: str = "1:1",
                       description_text: str = "") -> tuple:
    """Artwork (background) + shadow + scaled-down centred subject.

    Returns (result, subject_bottom_y) — the bottom edge feeds the caption
    positioning in add_description_text (state that used to ride session).
    """
    CS = base.width  # 1600
    bg_color = (253, 253, 242)
    canvas_img = Image.new("RGBA", (CS, CS), tuple(bg_color) + (255,))

    artwork_w, artwork_h = 0, 0
    if bg_image is not None:
        bw, bh = bg_image.size
        ar = _RATIO_AR.get(target_ratio, 1.0)
        if ar >= 1.0:
            safe_width, safe_height = CS, int(CS / ar)
        else:
            safe_width, safe_height = int(CS * ar), CS
        safe_margin = 0.02
        max_w = int(safe_width * (1.0 - 2 * safe_margin))
        max_h = int(safe_height * (1.0 - 2 * safe_margin))
        scale = min(max_w / float(bw), max_h / float(bh))
        artwork_w = max(1, int(bw * scale))
        artwork_h = max(1, int(bh * scale))

    subject_w, subject_h = 0, 0
    if subject_rgba is not None:
        scale_factor = 0.94
        subject_h, subject_w = subject_rgba.shape[:2]
        subject_w = max(1, int(subject_w * scale_factor))
        subject_h = max(1, int(subject_h * scale_factor))

    desc_height = 0
    if description_text.strip():
        font_size = max(32, int(CS * 0.04 * 0.4))
        line_height = int(font_size * 1.2)
        max_width = int(CS * 0.9)
        word_count = len(description_text.split())
        avg_chars_per_line = max(3, max_width // (font_size * 0.6))
        estimated_words_per_line = max(1, int(avg_chars_per_line / 5))
        estimated_lines = max(1, (word_count + estimated_words_per_line - 1)
                              // estimated_words_per_line)
        desc_height = int(line_height * estimated_lines + CS * 0.03)

    artwork_gap = int(CS * 0.03)
    subject_gap = int(CS * 0.03)
    total_comp_height = artwork_h + artwork_gap + subject_h + subject_gap + desc_height
    available_space = CS - total_comp_height
    vertical_offset = max(0, available_space // 2)

    artwork_bottom_y = 0
    if bg_image is not None:
        bw, bh = bg_image.size
        ar = _RATIO_AR.get(target_ratio, 1.0)
        if ar >= 1.0:
            safe_width, safe_height = CS, int(CS / ar)
        else:
            safe_width, safe_height = int(CS * ar), CS
        safe_margin = 0.02
        max_w = int(safe_width * (1.0 - 2 * safe_margin))
        max_h = int(safe_height * (1.0 - 2 * safe_margin))
        scale = min(max_w / float(bw), max_h / float(bh))
        new_w = max(1, int(bw * scale))
        new_h = max(1, int(bh * scale))
        try:
            resized = bg_image.resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            resized = bg_image.resize((new_w, new_h))
        rgba = resized.convert("RGBA")
        x = (CS - new_w) // 2
        y = vertical_offset + max(int(CS * 0.02), int(safe_height * safe_margin))
        canvas_img.paste(rgba, (x, y), rgba)
        artwork_bottom_y = y + new_h
    else:
        base_rgb = np.full((CS, CS, 3), bg_color, dtype=np.uint8)
        text_art = _render_background_text(Image.fromarray(base_rgb), CS)
        text_rgba = text_art.convert("RGBA")
        canvas_img.paste(text_rgba, (0, 0), text_rgba)

    canvas = np.array(canvas_img.convert("RGB"))

    if subject_rgba is None:
        result = Image.fromarray(canvas)
        result = ImageEnhance.Brightness(result).enhance(1.02)
        result = ImageEnhance.Contrast(result).enhance(1.05)
        return result, CS

    scale_factor = 0.94
    subject_h, subject_w = subject_rgba.shape[:2]
    new_w = max(1, int(subject_w * scale_factor))
    new_h = max(1, int(subject_h * scale_factor))

    scaled_rgb = cv2.resize(subject_rgba[:, :, :3], (new_w, new_h),
                            interpolation=cv2.INTER_LANCZOS4)
    scaled_alpha = cv2.resize(subject_rgba[:, :, 3], (new_w, new_h),
                              interpolation=cv2.INTER_LANCZOS4)

    center_x = CS // 2
    ox = center_x - new_w // 2

    if artwork_bottom_y > 0:
        oy = artwork_bottom_y + int(CS * 0.03)
    else:
        center_y = CS // 2
        oy = center_y - new_h // 2
        oy += vertical_offset - int(CS * 0.04)
    oy = max(0, min(oy, CS - new_h))

    cast_shadow, _ = generate_product_shadow(
        scaled_alpha, 0, 0, new_w, new_h, ox, oy, center_x, CS)

    total_shadow = np.clip(cast_shadow, 0.0, 1.0)
    canvas_f = canvas.astype(np.float32)
    for c in range(3):
        canvas_f[:, :, c] = np.clip(canvas_f[:, :, c] * (1.0 - total_shadow), 0, 255)
    canvas = canvas_f.astype(np.uint8)

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

    subject_bottom = can_y.stop if hasattr(can_y, "stop") else (oy + new_h)

    result = Image.fromarray(canvas)
    result = ImageEnhance.Brightness(result).enhance(1.02)
    result = ImageEnhance.Contrast(result).enhance(1.05)
    return result, subject_bottom


def add_description_text(img: Image.Image, text: str, subject_bottom_y: int) -> Image.Image:
    """Small grey caption under the product (bottom edge from apply_text_overlay)."""
    if not text.strip():
        return img

    W, H = img.size
    draw = ImageDraw.Draw(img)
    font_size = max(32, int(W * 0.04))
    font = _load_font(font_size, "Clean & Minimal")

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

    line_height = int(font_size * 1.2)
    total_height = line_height * len(lines)

    full_size = 1600
    scale_factor = H / float(full_size) if H < full_size else 1.0
    adjusted_subject_bottom = int(subject_bottom_y * scale_factor)

    gap = int(W * 0.03)
    y = min(adjusted_subject_bottom + gap, H - total_height - int(W * 0.02))
    y = max(y, int(W * 0.01))

    for line in lines:
        width = draw.textlength(line, font=font)
        x = (W - width) // 2
        draw.text((x, y), line, font=font, fill=(100, 100, 100, 255))
        y += line_height

    return img


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


def fit_to_ratio(img: Image.Image, ratio: str) -> Image.Image:
    """Pad img with its own background colour to the target aspect ratio."""
    W, H = img.size
    ar = _RATIO_AR.get(ratio, 1.0)
    cur = W / float(H)
    if abs(cur - ar) < 1e-3:
        return img
    if ar > cur:
        new_w, new_h = int(round(H * ar)), H
    else:
        new_w, new_h = W, int(round(W / ar))
    canvas = Image.new(img.mode, (new_w, new_h), _corner_color(img))
    canvas.paste(img, ((new_w - W) // 2, (new_h - H) // 2))
    return canvas


def compose(result: Image.Image, subject_rgba, ratio: str,
            text_mode: bool, caption: str) -> Image.Image:
    """Final image for one photo at its chosen ratio + text settings.

    Mirrors the Streamlit result-screen composition: build the 1:1 square
    (text overlay + caption), then pad out to the target ratio.
    """
    caption = caption if text_mode else ""
    if text_mode:
        square, subject_bottom = apply_text_overlay(
            result, subject_rgba, load_bg_image(), "1:1", caption)
        if caption.strip():
            square = add_description_text(square, caption.strip(), subject_bottom)
    else:
        square = result
    return fit_to_ratio(square, ratio)


def to_png_bytes(img: Image.Image, optimize: bool = False) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=optimize)
    return buf.getvalue()


def to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
