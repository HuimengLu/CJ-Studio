"""CJ Studio — Social Media Generator (V1).

A guided 4-step workflow that turns one uploaded photo into a branded social
graphic:

    1. Choose output ratio + upload   (upload auto-advances)
    2. Browse / select a template     (previews use the uploaded photo)
    3. Edit title + subtitle          (live preview)
    4. Review + download PNG

Templates replicate the CJ Studio Figma file (Funnnn, styles 1-7). Decoration
assets are exported from Figma at 4x into static/social/proc_*.png (see
scripts/prep_social_assets.py); each template is a declarative layout in a
270-wide design space so new templates/ratios/fields stay data-driven.

State lives in st.session_state["soc"]:
    step, ratio, img_bytes, img_hash, template, texts{slot: str}
Nothing is recomputed on back-navigation; renders are cached.
"""
import base64
import functools
import hashlib
import io
import os

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ── palette (matches app.py / Figma) ───────────────────────────────────────────
GREEN = (0, 86, 24)          # CJ brand green #005618
LIME = (188, 240, 14)        # scribble accent #BCF00E
INK = "#1b1b1b"

_BASE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_BASE, "static", "social")
_FONT_PATH = os.path.join(_BASE, "fonts", "Gabarito.ttf")

# ═══════════════════════════════════════════════════════════════════════════════
# Template engine
# ═══════════════════════════════════════════════════════════════════════════════
# All geometry is in "design units": the canvas is always 270 units wide and
# 270 / aspect-ratio tall (333 for 4:5 — the Figma master size — 270 for 1:1,
# 480 for 9:16). Rendering scales by S = out_w / 270.

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
    """Scale + centre-crop `img` to exactly (w, h) — never stretches."""
    return ImageOps.fit(img, (max(1, w), max(1, h)), Image.LANCZOS)


def _rounded_mask(w: int, h: int, r: int) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


def _paste_deco(canvas: Image.Image, name: str, x: float, y: float, w: float,
                S: float, opacity: float = 1.0) -> None:
    """Paste a 4x decoration asset scaled so its width is `w` design units."""
    a = _asset(name)
    tw = max(1, round(w * S))
    th = max(1, round(a.height * tw / a.width))
    a = a.resize((tw, th), Image.LANCZOS)
    if opacity < 1.0:
        alpha = a.getchannel("A").point(lambda v: int(v * opacity))
        a.putalpha(alpha)
    canvas.alpha_composite(a, (round(x * S), round(y * S)))


def _v_gradient(canvas: Image.Image, x: float, y: float, w: float, h: float,
                rgb: tuple, a0: int, a1: int, S: float, radius: float = 0) -> None:
    """Vertical alpha gradient rgb/a0 (top) → rgb/a1 (bottom)."""
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


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
          max_w: float, max_lines: int) -> list:
    """Greedy word-wrap; overflow beyond max_lines gets an ellipsis."""
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


def _text(canvas: Image.Image, text: str, x: float, y: float, size: float,
          lh: float, S: float, weight: str = "Bold", fill=(255, 255, 255, 255),
          max_w: float = 238, max_lines: int = 2, align: str = "left") -> float:
    """Draw wrapped text (Figma-style line boxes). Returns bottom y (design)."""
    if not text:
        return y
    draw = ImageDraw.Draw(canvas)
    font = _font(round(size * S), weight)
    lines = _wrap(draw, text, font, max_w * S, max_lines)
    ascent, descent = font.getmetrics()
    top_gap = (lh * S - (ascent + descent)) / 2  # glyph box centred in line box
    for i, line in enumerate(lines):
        ly = y * S + i * lh * S + top_gap
        lx = x * S
        if align == "right":
            lx = (x + max_w) * S - draw.textlength(line, font=font)
        elif align == "center":
            lx = (x + max_w / 2) * S - draw.textlength(line, font=font) / 2
        draw.text((lx, ly), line, font=font, fill=fill)
    return y + len(lines) * lh


# ── per-template renderers ─────────────────────────────────────────────────────
# Each receives (canvas RGBA, photo RGB, texts dict, H design height, S scale).
# Positions for H == 333 are Figma-exact; other ratios re-anchor sensibly.

def _r_style1(c, photo, texts, H, S):
    c.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    _paste_deco(c, "proc_s1_scribble.png", W - 141, -12, 156, S)
    _v_gradient(c, 0, H - 130, W, 130, GREEN, 0, 255, S)
    ty = _text(c, texts.get("title", ""), 16, H - 87, 26, 26, S)
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
    blob = _asset("proc_s3_blob.png")           # green blob; alpha = mask shape
    if H == 333:
        bs, bx, ty_top = 253, 0, 257             # Figma-exact
    else:                                        # 1:1 — compact + centre
        bs = H - 80
        bx = (W - bs - 17) / 2
        ty_top = H - 74
    bpx = round(bs * S)
    blob_s = blob.resize((bpx, bpx), Image.LANCZOS)
    c.alpha_composite(blob_s, (round(bx * S), 0))          # green backdrop blob
    ph = _cover(photo, bpx, bpx).convert("RGBA")
    ph.putalpha(blob_s.getchannel("A"))          # photo clipped to same blob
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

# Sample copy shown in gallery previews before the user edits text.
_SAMPLE_TEXTS = {"title": "Your Title Here",
                 "subtitle": "A short description of your post goes here"}


@st.cache_data(show_spinner=False, max_entries=96)
def render_template(tpl_id: str, ratio_id: str, img_bytes: bytes,
                    texts_key: tuple, out_w: int) -> bytes:
    """Render a template to PNG bytes. Cached on all inputs."""
    tpl = TEMPLATE_BY_ID[tpl_id]
    ratio = RATIO_BY_ID[ratio_id]
    H = ratio["H"]
    S = out_w / W
    texts = dict(texts_key)
    photo = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    canvas = Image.new("RGBA", (out_w, round(H * S)), (255, 255, 255, 255))
    tpl["render"](canvas, photo, texts, H, S)
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _texts_key(tpl: dict, texts: dict, samples: bool) -> tuple:
    """Hashable text tuple for a template's slots (placeholders if empty)."""
    out = []
    for slot in tpl["slots"]:
        v = (texts.get(slot["key"]) or "").strip()
        if not v and samples:
            v = _SAMPLE_TEXTS.get(slot["key"], "")
        out.append((slot["key"], v))
    return tuple(out)


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """
/* ── social module shell ── */
[data-testid="stHeader"]{display:none !important}
.block-container{max-width:1180px !important;padding-top:0 !important;
  padding-bottom:0 !important}
/* fixed screen title (back arrow + serif heading) */
#cj-soc-head{position:fixed;top:0;left:280px;right:0;z-index:40;display:flex;
  align-items:center;gap:18px;padding:26px 48px 18px;pointer-events:none;
  background:linear-gradient(#f8f9fa 62%,rgba(248,249,250,0))}
#cj-soc-head .cj-soc-title{font-family:'Libre Caslon Text',Georgia,serif;
  font-style:italic;font-size:28px;color:#111;line-height:1.2}
#cj-soc-head .cj-soc-step{font-family:'Hanken Grotesk',sans-serif;font-size:11px;
  font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#9a9a9a;
  margin-left:auto}
/* back arrow = st.button pinned into the header row */
.st-key-soc_back{position:fixed;top:22px;left:300px;z-index:41;width:44px}
.st-key-soc_back button{width:44px !important;height:44px !important;
  border-radius:50% !important;background:transparent !important;
  border:none !important;color:#111 !important;font-size:22px !important;
  padding:0 !important;display:flex;align-items:center;justify-content:center}
.st-key-soc_back button:hover{background:#e7e8e9 !important;color:#000 !important}
.st-key-soc_back button p{font-size:22px !important;line-height:1}
/* arc progress indicator */
#cj-arc{position:fixed;left:300px;top:50%;transform:translateY(-50%);
  z-index:30;pointer-events:none}
#cj-arc path{fill:none;stroke:#e5e7eb;stroke-width:1}
#cj-arc circle{transition:all .3s ease}
/* snap carousel */
[data-testid="stMain"]{scroll-snap-type:y mandatory}
[class*="st-key-soc_card_"]{scroll-snap-align:center;
  transition:transform .5s cubic-bezier(.25,1,.5,1),opacity .5s ease;
  transform:scale(.92);opacity:.45}
[class*="st-key-soc_card_"].cj-active{transform:scale(1);opacity:1}
[data-testid="stVerticalBlockBorderWrapper"]:has(>[class*="st-key-soc_card_"]){
  border:none !important;background:transparent !important;padding:0 !important}
[class*="st-key-soc_card_"]{gap:0 !important}
.cj-snap-pad{height:20vh}
/* card label under each frame */
.cj-card-lbl{text-align:center;margin-top:26px}
.cj-card-lbl h2{font-family:'Libre Caslon Text',Georgia,serif;font-size:26px;
  font-weight:400;color:#1b1b1b;margin:0}
.cj-card-lbl p{font-family:'Hanken Grotesk',sans-serif;font-size:13px;
  font-weight:600;letter-spacing:.08em;color:#5d5e66;margin:6px 0 0}
/* ── step 1: ratio cards double as uploaders ── */
[data-testid="stFileUploader"]{border:none !important;background:transparent
  !important;padding:0 !important;margin:0 !important}
[data-testid="stFileUploader"] > label{display:none}
section[data-testid="stFileUploaderDropzone"]{position:relative;margin:0 auto;
  padding:0 !important;border:1px solid #e7e8e9;border-radius:2px;
  background:#fff;box-shadow:0 12px 32px rgba(0,0,0,.05);
  display:flex !important;align-items:center !important;
  justify-content:center !important;overflow:hidden;
  transition:border-color .3s,box-shadow .3s}
section[data-testid="stFileUploaderDropzone"]::before{content:"";position:absolute;
  inset:8px;background:#f3f4f5;
  background-image:radial-gradient(#e5e7eb 1px,transparent 1px);
  background-size:16px 16px}
section[data-testid="stFileUploaderDropzone"]:hover{border-color:#1b1b1b}
[data-testid="stFileUploaderDropzoneInstructions"]{z-index:1;display:flex
  !important;flex-direction:column !important;align-items:center !important;
  margin:0 !important}
[data-testid="stFileUploaderDropzoneInstructions"] > *{display:none !important}
[data-testid="stFileUploaderDropzoneInstructions"]::before{
  font-family:'Material Symbols Outlined';content:"\\e439";font-size:38px;
  color:#5d5e66;transition:transform .3s,color .3s}
section[data-testid="stFileUploaderDropzone"]:hover
  [data-testid="stFileUploaderDropzoneInstructions"]::before{
  transform:scale(1.12);color:#1b1b1b}
section[data-testid="stFileUploaderDropzone"] > button{position:absolute
  !important;inset:0;width:100% !important;height:100% !important;margin:0
  !important;padding:0 !important;border:none !important;background:transparent
  !important;color:transparent !important;opacity:0;cursor:pointer;z-index:3}
[class*="st-key-soc_up_1x1"] section[data-testid="stFileUploaderDropzone"]{
  width:400px;height:400px}
[class*="st-key-soc_up_4x5"] section[data-testid="stFileUploaderDropzone"]{
  width:360px;height:444px}
[class*="st-key-soc_up_9x16"] section[data-testid="stFileUploaderDropzone"]{
  width:270px;height:480px}
/* ── step 2: template cards ── */
[class*="st-key-soc_card_t_"]{position:relative}
.cj-tpl-frame{background:#fff;border:1px solid #e7e8e9;border-radius:2px;
  padding:8px;box-shadow:0 12px 32px rgba(0,0,0,.05);display:inline-block;
  transition:border-color .3s,box-shadow .3s}
.cj-tpl-frame img{display:block;width:100%}
.cj-tpl-wrap{text-align:center}
[class*="st-key-soc_card_t_"].cj-selected .cj-tpl-frame{border:2px solid #1b1b1b;
  padding:7px;box-shadow:0 12px 32px rgba(0,0,0,.12)}
[class*="st-key-soc_card_t_"].cj-selected .cj-card-lbl p::after{
  content:"  ·  SELECTED";color:#1b1b1b}
/* invisible pick button stretched over the card */
[class*="st-key-soc_card_t_"] [data-testid="stElementContainer"]:has(
  [class*="st-key-soc_pick_"]),
[class*="st-key-soc_pick_"],[class*="st-key-soc_pick_"] button{
  position:absolute !important;inset:0 !important;z-index:5;margin:0 !important}
[class*="st-key-soc_pick_"] button{width:100% !important;height:100% !important;
  opacity:0 !important;cursor:pointer;background:transparent !important;
  border:none !important}
/* ── fixed bottom action bar (Next / Download) ── */
.st-key-soc_actions{position:fixed;bottom:0;left:280px;right:0;z-index:40;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:18px 48px 26px;
  background:linear-gradient(rgba(248,249,250,0),#f8f9fa 55%)}
.st-key-soc_actions [data-testid="stElementContainer"]{width:400px;margin:0}
.st-key-soc_actions button,.st-key-soc_actions a{width:100% !important;
  background:#111 !important;color:#fff !important;border:none !important;
  border-radius:0 !important;padding:15px 28px !important;
  font-family:'Hanken Grotesk',sans-serif !important;font-size:15px !important;
  font-weight:500 !important;letter-spacing:.01em}
.st-key-soc_actions button:hover,.st-key-soc_actions a:hover{
  background:#000 !important;color:#fff !important}
.st-key-soc_actions button:disabled{background:#c9cbcd !important;
  color:#f3f4f5 !important;cursor:not-allowed}
.st-key-soc_actions button p{font-size:15px !important}
/* ── step 3: edit content ── */
.st-key-soc_edit_row{padding-top:110px}
.cj-preview-card{background:#fff;border:1px solid #e7e8e9;border-radius:2px;
  padding:8px;box-shadow:0 12px 32px rgba(0,0,0,.05);display:inline-block}
.cj-preview-card img{display:block;width:100%}
.cj-field-lbl{font-family:'Hanken Grotesk',sans-serif;font-size:12px;
  font-weight:500;letter-spacing:.04em;text-transform:uppercase;color:#9ca3af;
  margin:34px 0 2px}
.cj-field-lbl:first-child{margin-top:0}
[class*="st-key-soc_txt_"] [data-baseweb="input"]{background:transparent
  !important;border:none !important;border-bottom:1px solid #d1d5db !important;
  border-radius:0 !important}
[class*="st-key-soc_txt_"] [data-baseweb="base-input"]{background:transparent
  !important;border:none !important}
[class*="st-key-soc_txt_"] [data-baseweb="input"]:focus-within{
  border-bottom-color:#111 !important}
[class*="st-key-soc_txt_"] input{font-family:'Hanken Grotesk',sans-serif
  !important;font-size:20px !important;font-weight:500 !important;color:#111
  !important;padding:8px 0 12px !important;background:transparent !important}
[class*="st-key-soc_txt_"] input::placeholder{font-style:italic;
  color:#d1d5db !important}
.cj-noslots{font-family:'Hanken Grotesk',sans-serif;font-size:14px;
  color:#5d5e66;font-style:italic;margin-top:8px}
/* ── step 4: review ── */
.cj-review-cap{font-family:'Hanken Grotesk',sans-serif;font-size:16px;
  color:#9ca3af;margin:14px 0 0;text-align:left}
.cj-review-col{display:flex;flex-direction:column;align-items:center;
  padding:110px 0 130px}
@media (max-width:900px){
  #cj-soc-head{left:0}
  .st-key-soc_back{left:8px}
  #cj-arc{display:none}
  .st-key-soc_actions{left:0}
}
"""

# Arc indicator geometry lifted from the approved carousel reference.
def _arc_html(n: int, active: int) -> str:
    dots = []
    for i in range(n):
        cy = 400 * (i + 1) / (n + 1)
        r, cx, fill = (6, 60, "#1b1b1b") if i == active else (4, 28, "#d1d5db")
        dots.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"></circle>')
    return (
        '<svg id="cj-arc" width="100" height="400" viewBox="0 0 100 400" '
        'style="overflow:visible">'
        '<path d="M 0,0 C 80,100 80,300 0,400"></path>' + "".join(dots) + "</svg>"
    )


# Tiny same-origin iframe that wires scroll-snap active states + arc dots on
# the parent document (st.markdown strips <script>).
_CAROUSEL_JS = """
<script>
(function(){
  var P;
  try { P = window.parent.document; } catch(e) { return; }
  var tries = 0;
  function init(){
    var main = P.querySelector('section[data-testid="stMain"]');
    var cards = Array.prototype.slice.call(
        P.querySelectorAll('[class*="st-key-soc_card_"]'));
    if((!main || !cards.length) && tries++ < 40){ setTimeout(init, 150); return; }
    if(!main || !cards.length) return;
    var dots = Array.prototype.slice.call(P.querySelectorAll('#cj-arc circle'));
    function upd(){
      var mr = main.getBoundingClientRect();
      var mc = mr.top + main.clientHeight / 2;
      var best = 0, bd = 1e9;
      cards.forEach(function(c, i){
        var r = c.getBoundingClientRect();
        var d = Math.abs(r.top + r.height/2 - mc);
        if(d < bd){ bd = d; best = i; }
      });
      cards.forEach(function(c, i){
        c.classList.toggle('cj-active', i === best);
      });
      dots.forEach(function(d, i){
        d.setAttribute('r', i === best ? '6' : '4');
        d.setAttribute('cx', i === best ? '60' : '28');
        d.setAttribute('fill', i === best ? '#1b1b1b' : '#d1d5db');
      });
    }
    main.addEventListener('scroll', upd, {passive:true});
    upd();
    // centre the pre-selected card (step 2 keeps selection on back-nav)
    var sel = P.querySelector('[class*="st-key-soc_card_"].cj-selected');
    if(sel) sel.scrollIntoView({block:'center'});
  }
  init();
})();
</script>
"""


def _init_state() -> dict:
    if "soc" not in st.session_state:
        st.session_state.soc = {
            "step": 1, "ratio": None, "img_bytes": None, "img_hash": None,
            "template": None, "texts": {}, "up_nonce": 0,
        }
    return st.session_state.soc


def _head(title: str, step: int, back_to: int | None) -> None:
    st.markdown(
        f"<div id='cj-soc-head'><span style='width:44px'></span>"
        f"<span class='cj-soc-title'>{title}</span>"
        f"<span class='cj-soc-step'>Step {step} / 4</span></div>",
        unsafe_allow_html=True,
    )
    if back_to is not None:
        def _go_back():
            soc = st.session_state.soc
            soc["step"] = back_to
            if back_to == 1:
                # fresh uploader widgets, or their retained value would
                # immediately auto-advance right back to step 2
                soc["up_nonce"] = soc.get("up_nonce", 0) + 1
        st.button("←", key="soc_back", on_click=_go_back)


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


# ── Step 1: output ratio + upload ──────────────────────────────────────────────

def _step1(soc: dict) -> None:
    _head("Upload an image", 1, None)
    st.markdown(_arc_html(len(RATIOS), 1), unsafe_allow_html=True)
    st.markdown("<div class='cj-snap-pad'></div>", unsafe_allow_html=True)

    for ratio in RATIOS:
        rid = ratio["id"]
        key = rid.replace(":", "x")
        with st.container(key=f"soc_card_{key}"):
            up = st.file_uploader(
                f"Upload {ratio['name']}",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"soc_up_{key}_{soc.get('up_nonce', 0)}",
                label_visibility="collapsed",
            )
            st.markdown(
                f"<div class='cj-card-lbl'><h2>{ratio['name']}</h2>"
                f"<p>{rid}</p></div>",
                unsafe_allow_html=True,
            )
        if up is not None:
            raw = up.getvalue()
            soc.update(
                ratio=rid, img_bytes=raw,
                img_hash=hashlib.sha1(raw).hexdigest(),
                step=2,
            )
            # keep the current template only if it fits the new ratio
            tpl = soc.get("template")
            if tpl and rid not in TEMPLATE_BY_ID[tpl]["ratios"]:
                soc["template"] = None
            st.rerun()

    st.markdown("<div class='cj-snap-pad'></div>", unsafe_allow_html=True)

    # back-navigation case: an image is already uploaded, allow continuing
    # without re-uploading (state is preserved per the PRD)
    if soc["img_bytes"] is not None:
        with st.container(key="soc_actions"):
            def _cont():
                st.session_state.soc["step"] = 2
            st.button("Continue  →", key="soc_next1", on_click=_cont,
                      use_container_width=True)
    components.html(_CAROUSEL_JS, height=1)


# ── Step 2: template gallery ───────────────────────────────────────────────────

def _step2(soc: dict) -> None:
    _head("Select A Layout", 2, 1)
    tpls = [t for t in TEMPLATES if soc["ratio"] in t["ratios"]]
    sel_idx = next((i for i, t in enumerate(tpls) if t["id"] == soc["template"]), 0)
    st.markdown(_arc_html(len(tpls), sel_idx), unsafe_allow_html=True)
    st.markdown("<div class='cj-snap-pad'></div>", unsafe_allow_html=True)

    ratio = RATIO_BY_ID[soc["ratio"]]
    card_w = {"1:1": 400, "4:5": 360, "9:16": 270}[soc["ratio"]]

    for tpl in tpls:
        png = render_template(
            tpl["id"], soc["ratio"], soc["img_bytes"],
            _texts_key(tpl, soc["texts"], samples=True), 540,
        )
        with st.container(key=f"soc_card_t_{tpl['id']}"):
            st.markdown(
                f"<div class='cj-tpl-wrap'>"
                f"<div class='cj-tpl-frame' style='width:{card_w}px'>"
                f"<img src='{_data_url(png)}' alt='{tpl['name']}'/></div>"
                f"<div class='cj-card-lbl'><h2>{tpl['name']}</h2>"
                f"<p>{ratio['name']} · {soc['ratio']}</p></div></div>",
                unsafe_allow_html=True,
            )

            def _pick(tid=tpl["id"]):
                st.session_state.soc["template"] = tid
            st.button(" ", key=f"soc_pick_{tpl['id']}", on_click=_pick)

    st.markdown("<div class='cj-snap-pad'></div>", unsafe_allow_html=True)

    # selected-state styling rides on the container's stable st-key class
    if soc["template"]:
        st.markdown(
            f"<style>.st-key-soc_card_t_{soc['template']}{{opacity:1}}"
            f".st-key-soc_card_t_{soc['template']} .cj-tpl-frame{{"
            f"border:2px solid #1b1b1b;padding:7px}}"
            f".st-key-soc_card_t_{soc['template']} .cj-card-lbl p::after{{"
            f"content:'  ·  SELECTED';color:#1b1b1b}}</style>",
            unsafe_allow_html=True,
        )

    with st.container(key="soc_actions"):
        def _next():
            st.session_state.soc["step"] = 3
        st.button("Next  →", key="soc_next2", on_click=_next,
                  disabled=soc["template"] is None,
                  use_container_width=True)
    components.html(_CAROUSEL_JS, height=1)


# ── Step 3: edit content (live preview) ────────────────────────────────────────

def _step3(soc: dict) -> None:
    _head("Edit Content", 3, 2)
    tpl = TEMPLATE_BY_ID[soc["template"]]

    with st.container(key="soc_edit_row"):
        prev_col, form_col = st.columns([11, 9], gap="large")

        with form_col:
            if tpl["slots"]:
                for slot in tpl["slots"]:
                    st.markdown(
                        f"<p class='cj-field-lbl'>{slot['label']}</p>",
                        unsafe_allow_html=True,
                    )
                    val = st.text_input(
                        slot["label"],
                        value=soc["texts"].get(slot["key"], ""),
                        placeholder=slot["placeholder"],
                        key=f"soc_txt_{slot['key']}",
                        label_visibility="collapsed",
                    )
                    soc["texts"][slot["key"]] = val
            else:
                st.markdown(
                    "<p class='cj-noslots'>This template has no editable "
                    "text — it lets your photo do the talking.</p>",
                    unsafe_allow_html=True,
                )

        with prev_col:
            # sample copy stands in for empty fields (matches the Figma
            # screens); the final render in step 4 uses only real text.
            png = render_template(
                tpl["id"], soc["ratio"], soc["img_bytes"],
                _texts_key(tpl, soc["texts"], samples=True), 720,
            )
            pw = {"1:1": 470, "4:5": 440, "9:16": 330}[soc["ratio"]]
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div class='cj-preview-card' style='width:{pw}px'>"
                f"<img src='{_data_url(png)}'/></div></div>",
                unsafe_allow_html=True,
            )

    with st.container(key="soc_actions"):
        def _next():
            st.session_state.soc["step"] = 4
        st.button("Next  →", key="soc_next3", on_click=_next,
                  use_container_width=True)


# ── Step 4: review + download ──────────────────────────────────────────────────

def _step4(soc: dict) -> None:
    _head("Review Your Post", 4, 3)
    tpl = TEMPLATE_BY_ID[soc["template"]]
    ratio = RATIO_BY_ID[soc["ratio"]]

    png = render_template(
        tpl["id"], soc["ratio"], soc["img_bytes"],
        _texts_key(tpl, soc["texts"], samples=False), ratio["out"][0],
    )
    pw = {"1:1": 480, "4:5": 440, "9:16": 320}[soc["ratio"]]
    st.markdown(
        f"<div class='cj-review-col'>"
        f"<div><div class='cj-preview-card' style='width:{pw}px'>"
        f"<img src='{_data_url(png)}'/></div>"
        f"<p class='cj-review-cap'>{ratio['name']} {soc['ratio']}</p></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.container(key="soc_actions"):
        st.download_button(
            "Download  ↓",
            data=png,
            file_name=f"cj_social_{tpl['id']}_{soc['ratio'].replace(':', 'x')}.png",
            mime="image/png",
            key="soc_dl",
            use_container_width=True,
        )


def render() -> None:
    """Entry point — called from app.py when ?view=social."""
    soc = _init_state()
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

    step = soc["step"]
    # guard: refresh drops nothing, but a direct deep-link may lack state
    if step >= 2 and soc["img_bytes"] is None:
        step = soc["step"] = 1
    if step >= 3 and soc["template"] is None:
        step = soc["step"] = 2

    {1: _step1, 2: _step2, 3: _step3, 4: _step4}[step](soc)
