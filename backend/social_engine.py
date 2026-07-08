"""Social template engine — extracted from social.py, framework-free.

Templates replicate the CJ Studio Figma file (Funnnn, styles 1-7) as
declarative layouts in a 270-wide design space; decoration assets are 4x
Figma exports in static/social/proc_*.png (see scripts/prep_social_assets.py).

    render_template(tpl_id, ratio_id, img_bytes, texts, out_w) -> PNG bytes

Results are memoised in a small LRU keyed on all inputs (image by sha1).
"""
import functools
import hashlib
import io
import os
from collections import OrderedDict

from PIL import Image, ImageDraw, ImageFont, ImageOps

GREEN = (0, 86, 24)          # CJ brand green #005618
LIME = (188, 240, 14)        # scribble accent #BCF00E

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS = os.path.join(_BASE, "static", "social")
_FONT_PATH = os.path.join(_BASE, "fonts", "Gabarito.ttf")

# Design space: canvas is always 270 units wide; height per ratio (333 for 4:5
# — the Figma master size — 270 for 1:1, 480 for 9:16). S = out_w / 270.
RATIOS = [
    {"id": "1:1",  "name": "Facebook Post",   "H": 270, "out": (1080, 1080)},
    {"id": "4:5",  "name": "Instagram Post",  "H": 333, "out": (1080, 1332)},
    {"id": "9:16", "name": "Instagram Story", "H": 480, "out": (1080, 1920)},
]
RATIO_BY_ID = {r["id"]: r for r in RATIOS}

W = 270  # design-space width, always


@functools.lru_cache(maxsize=32)
def _asset(name: str) -> Image.Image:
    return Image.open(os.path.join(_ASSETS, name)).convert("RGBA")


@functools.lru_cache(maxsize=64)
def _font(size_px: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(_FONT_PATH, size_px)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    return ImageOps.fit(img, (max(1, w), max(1, h)), Image.LANCZOS)


def _rounded_mask(w: int, h: int, r: int) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


def _paste_deco(canvas, name, x, y, w, S, opacity: float = 1.0) -> None:
    a = _asset(name)
    tw = max(1, round(w * S))
    th = max(1, round(a.height * tw / a.width))
    a = a.resize((tw, th), Image.LANCZOS)
    if opacity < 1.0:
        alpha = a.getchannel("A").point(lambda v: int(v * opacity))
        a.putalpha(alpha)
    canvas.alpha_composite(a, (round(x * S), round(y * S)))


def _v_gradient(canvas, x, y, w, h, rgb, a0, a1, S, radius: float = 0) -> None:
    gw, gh = max(1, round(w * S)), max(1, round(h * S))
    grad = Image.new("L", (1, gh))
    grad.putdata([int(a0 + (a1 - a0) * i / max(1, gh - 1)) for i in range(gh)])
    grad = grad.resize((gw, gh))
    if radius:
        m = _rounded_mask(gw, gh, round(radius * S))
        grad = Image.composite(grad, Image.new("L", (gw, gh), 0), m)
    layer = Image.new("RGBA", (gw, gh), rgb + (255,))
    layer.putalpha(grad)
    canvas.alpha_composite(layer, (round(x * S), round(y * S)))


def _wrap(draw, text, font, max_w, max_lines) -> list:
    words, lines, cur = text.split(), [], ""
    for word in words:
        cand = (cur + " " + word).strip()
        if not cur or draw.textlength(cand, font=font) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + "…", font=font) > max_w:
            lines[-1] = lines[-1][:-1].rstrip()
        lines[-1] += "…"
    return lines


def _text(canvas, text, x, y, size, lh, S, weight="Bold", fill=(255, 255, 255, 255),
          max_w=238, max_lines=2, align="left") -> float:
    if not text:
        return y
    draw = ImageDraw.Draw(canvas)
    font = _font(round(size * S), weight)
    lines = _wrap(draw, text, font, max_w * S, max_lines)
    ascent, descent = font.getmetrics()
    top_gap = (lh * S - (ascent + descent)) / 2
    for i, line in enumerate(lines):
        ly = y * S + i * lh * S + top_gap
        lx = x * S
        if align == "right":
            lx = (x + max_w) * S - draw.textlength(line, font=font)
        elif align == "center":
            lx = (x + max_w / 2) * S - draw.textlength(line, font=font) / 2
        draw.text((lx, ly), line, font=font, fill=fill)
    return y + len(lines) * lh


# ── per-template renderers (positions Figma-exact at H=333) ────────────────────

def _r_style1(c, photo, texts, H, S):
    c.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    _paste_deco(c, "proc_s1_scribble.png", W - 141, -12, 156, S)
    _v_gradient(c, 0, H - 130, W, 130, GREEN, 0, 255, S)
    _text(c, texts.get("title", ""), 16, H - 87, 26, 26, S)
    _text(c, texts.get("subtitle", ""), 16, H - 31, 8, 10, S,
          weight="Regular", max_w=224)


def _r_style2(c, photo, texts, H, S):
    ImageDraw.Draw(c).rectangle([0, 0, W * S, H * S], fill=GREEN + (255,))
    ty = _text(c, texts.get("title", ""), 15, 14, 26, 26, S, max_w=240)
    _paste_deco(c, "proc_s2_scribble.png", 0, ty - 11, 162, S)
    _text(c, texts.get("subtitle", ""), 15, ty + 19, 8, 10, S,
          weight="Regular", max_w=224)
    iw, ih = round(250 * S), round((H - 128) * S)
    img = _cover(photo, iw, ih)
    c.paste(img, (round(10 * S), round(118 * S)),
            _rounded_mask(iw, ih, round(5 * S)))
    _v_gradient(c, 10, H - 76, 250, 66, (0, 0, 0), 0, 178, S, radius=5)


def _r_style3(c, photo, texts, H, S):
    ImageDraw.Draw(c).rectangle([0, 0, W * S, H * S], fill=(255, 255, 255, 255))
    blob = _asset("proc_s3_blob.png")
    if H == 333:
        bs, bx, ty_top = 253, 0, 257             # Figma-exact
    else:                                        # 1:1 — compact + centre
        bs = H - 80
        bx = (W - bs - 17) / 2
        ty_top = H - 74
    bpx = round(bs * S)
    blob_s = blob.resize((bpx, bpx), Image.LANCZOS)
    c.alpha_composite(blob_s, (round(bx * S), 0))
    ph = _cover(photo, bpx, bpx).convert("RGBA")
    ph.putalpha(blob_s.getchannel("A"))
    c.alpha_composite(ph, (round((bx + 17) * S), round(4 * S)))
    _paste_deco(c, "proc_s3_scribble.png", -6, -14, 99, S)
    ty = _text(c, texts.get("title", ""), 12, ty_top, 26, 26, S,
               fill=GREEN + (255,), max_w=246, align="center")
    _text(c, texts.get("subtitle", ""), 12, ty + 2, 8, 10, S, weight="Regular",
          fill=(0, 0, 0, 153), max_w=246, max_lines=1, align="center")


def _r_style4(c, photo, texts, H, S):
    c.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    panel = _asset("proc_s4_panel.png").resize(
        (round((W - 19) * S), round((H - 23) * S)), Image.LANCZOS)
    c.alpha_composite(panel, (round(9 * S), round(11 * S)))
    _paste_deco(c, "proc_s4_scribble.png", W - 128, 46, 121.7, S)
    ty = _text(c, texts.get("title", ""), 14, 16, 26, 26, S, max_w=202)
    _text(c, texts.get("subtitle", ""), 14, ty + 4, 8, 10, S,
          weight="Regular", fill=(255, 255, 255, 204), max_w=134, max_lines=3)


def _r_style5(c, photo, texts, H, S):
    c.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    frame = _asset("proc_s5_frame.png").resize(
        (round((W - 19) * S), round((H - 23) * S)), Image.LANCZOS)
    c.alpha_composite(frame, (round(9 * S), round(11 * S)))
    _paste_deco(c, "proc_s5_scribble.png", W - 105, H - 143, 101, S)
    ty = _text(c, texts.get("title", ""), 19, H - 86, 26, 26, S,
               fill=GREEN + (255,), max_w=231)
    _text(c, texts.get("subtitle", ""), 19, H - 32, 8, 10, S, weight="Regular",
          fill=(0, 0, 0, 153), max_w=231, max_lines=1, align="right")


def _dots_logo(c, photo, H, S, dots_name, opacity):
    c.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    _paste_deco(c, dots_name, W * 0.0615, H * 0.0620, W * 0.8770, S,
                opacity=opacity)
    _paste_deco(c, "proc_logo_white.png", W * 0.8567, H * 0.8950, W * 0.12, S)


def _r_style6(c, photo, texts, H, S):
    _dots_logo(c, photo, H, S, "s6_dots.png", 1.0)


def _r_style7(c, photo, texts, H, S):
    _dots_logo(c, photo, H, S, "s7_dots.png", 0.8)


# Text slots are declarative so future templates can add more (or zero) fields.
_SLOTS = [
    {"key": "title", "label": "Title", "placeholder": "Pittsburgh Penguins"},
    {"key": "subtitle", "label": "Description", "placeholder": "Hello World!"},
]

TEMPLATES = [
    {"id": "style1", "name": "Spotlight", "render": _r_style1,
     "ratios": ["1:1", "4:5", "9:16"], "slots": _SLOTS},
    {"id": "style2", "name": "Bulletin", "render": _r_style2,
     "ratios": ["1:1", "4:5", "9:16"], "slots": _SLOTS},
    {"id": "style3", "name": "Bloom", "render": _r_style3,
     "ratios": ["1:1", "4:5"], "slots": _SLOTS},
    {"id": "style4", "name": "Field Notes", "render": _r_style4,
     "ratios": ["4:5", "9:16"], "slots": _SLOTS},
    {"id": "style5", "name": "Gallery Frame", "render": _r_style5,
     "ratios": ["4:5", "9:16"], "slots": _SLOTS},
    {"id": "style6", "name": "Confetti", "render": _r_style6,
     "ratios": ["1:1"], "slots": []},
    {"id": "style7", "name": "Dot Grid", "render": _r_style7,
     "ratios": ["1:1"], "slots": []},
]
TEMPLATE_BY_ID = {t["id"]: t for t in TEMPLATES}


# ── render + cache ─────────────────────────────────────────────────────────────

_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_CACHE_MAX = 96


def render_template(tpl_id: str, ratio_id: str, img_bytes: bytes,
                    texts: dict, out_w: int) -> bytes:
    """Render a template to PNG bytes. Memoised on all inputs."""
    tpl = TEMPLATE_BY_ID[tpl_id]
    ratio = RATIO_BY_ID[ratio_id]
    texts = {k: (v or "").strip() for k, v in (texts or {}).items()}

    key = (tpl_id, ratio_id, hashlib.sha1(img_bytes).hexdigest(),
           tuple(sorted(texts.items())), out_w)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    H = ratio["H"]
    S = out_w / W
    photo = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    canvas = Image.new("RGBA", (out_w, round(H * S)), (255, 255, 255, 255))
    tpl["render"](canvas, photo, texts, H, S)
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    png = buf.getvalue()

    _CACHE[key] = png
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return png


def templates_for(ratio_id: str) -> list:
    """JSON-safe template list for one ratio (render fns stripped)."""
    return [
        {"id": t["id"], "name": t["name"], "slots": t["slots"]}
        for t in TEMPLATES if ratio_id in t["ratios"]
    ]
