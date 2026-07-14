"""Social template engine — extracted from social.py, framework-free.

Templates replicate the CJ Studio Figma file (Funnnn, styles 1-7) as
declarative layouts in a 270-wide design space; decoration assets are 4x
Figma exports in static/social/proc_*.png (see scripts/prep_social_assets.py).

    render_template(tpl_id, ratio_id, img_bytes, texts, out_w) -> PNG bytes
    recommend(has_image=…, has_title=…, has_subtitle=…, ratio_id=…) -> [rec]

Every template is declared as metadata (see TEMPLATES) so the recommendation
engine reasons purely over data — compatibility, layout variants and colour
themes — instead of hardcoded conditionals. Results are memoised in a small
LRU keyed on all inputs (image by sha1).
"""
import functools
import hashlib
import io
import os
import random
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
    """Style 1 "Spotlight" — Figma 266:547 card layout.

    White card: photo on top (10px margins, rounded 5, soft dark bottom
    gradient), bold #4c4c4c title (1-2 lines), lime scribble under the last
    title line, optional small subtitle (omitted entirely when empty).
    Text anchors from the card's bottom edge — subtitle top H-30, scribble
    top H-57, last title line H-68 — which reproduces every Figma variant
    (ratio x line-count); the photo absorbs the remaining height.
    """
    draw = ImageDraw.Draw(c)
    draw.rectangle([0, 0, W * S, H * S], fill=(255, 255, 255, 255))

    title = (texts.get("title") or "").strip() or "Construction Junction"
    subtitle = (texts.get("subtitle") or "").strip()

    # Measure the title first — the photo grows when the title is one line.
    font = _font(round(26 * S), "Bold")
    n_lines = len(_wrap(draw, title, font, 240 * S, 2))
    last_line_top = H - 68
    title_top = last_line_top - 26 * (n_lines - 1)
    photo_h = title_top - 25          # 10 margin + 15 gap above the title

    iw, ih = round(250 * S), round(photo_h * S)
    img = _cover(photo, iw, ih)
    c.paste(img, (round(10 * S), round(10 * S)),
            _rounded_mask(iw, ih, round(5 * S)))
    _v_gradient(c, 10, 10 + photo_h - 53, 250, 53, (0, 0, 0), 0, 178, S,
                radius=5)

    _text(c, title, 15, title_top, 26, 26, S,
          fill=(0x4C, 0x4C, 0x4C, 255), max_w=240)
    _paste_deco(c, "proc_s2_scribble.png", 7, H - 57, 161.8, S)
    if subtitle:
        _text(c, subtitle, 15, H - 30, 8, 10, S, weight="Regular",
              fill=(0x4C, 0x4C, 0x4C, 255), max_w=240, max_lines=2)


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
# Both are optional: an empty title falls back to the brand placeholder at
# render time; an empty subtitle simply doesn't appear.
_SLOTS = [
    {"key": "title", "label": "Title", "placeholder": "Construction Junction"},
    {"key": "subtitle", "label": "Description (optional)", "placeholder": ""},
]

# Colour themes a logical template can be rendered in. The renderers are still
# single-palette (Figma-exact); `theme` is threaded through the engine, API and
# cache key so themed rendering is a metadata-only extension later on.
THEMES = ["green", "lime", "white"]

# ── Template library (data-driven) ─────────────────────────────────────────────
# Each template is pure metadata + a renderer. The recommendation engine reads
# only the metadata, so new templates/variants/themes need no engine changes:
#   supportsImage    — needs a photo (all current templates do)
#   supportsTitle    — has a title slot        → an "image + text" template
#   supportsSubtitle — can show a subtitle line
#   maxTitleLines    — auto-fit ceiling for the title
#   maxSubtitleLines — auto-fit ceiling for the subtitle
#   themes           — colour variants this template supports
def _meta(supports_title, supports_subtitle, max_title, max_sub, themes):
    return {
        "supportsImage": True,
        "supportsTitle": supports_title,
        "supportsSubtitle": supports_subtitle,
        "maxTitleLines": max_title,
        "maxSubtitleLines": max_sub,
        "themes": themes,
    }

_TEXT_THEMES = ["green", "lime", "white"]
_IMG_THEMES = ["white"]

TEMPLATES = [
    {"id": "style1", "name": "Spotlight", "render": _r_style1,
     "ratios": ["1:1", "4:5", "9:16"], "slots": _SLOTS,
     **_meta(True, True, 2, 2, _TEXT_THEMES)},
    {"id": "style2", "name": "Bulletin", "render": _r_style2,
     "ratios": ["1:1", "4:5", "9:16"], "slots": _SLOTS,
     **_meta(True, True, 2, 2, _TEXT_THEMES)},
    {"id": "style3", "name": "Bloom", "render": _r_style3,
     "ratios": ["1:1", "4:5"], "slots": _SLOTS,
     **_meta(True, True, 2, 1, _TEXT_THEMES)},
    {"id": "style4", "name": "Field Notes", "render": _r_style4,
     "ratios": ["4:5", "9:16"], "slots": _SLOTS,
     **_meta(True, True, 2, 3, _TEXT_THEMES)},
    {"id": "style5", "name": "Gallery Frame", "render": _r_style5,
     "ratios": ["4:5", "9:16"], "slots": _SLOTS,
     **_meta(True, True, 2, 1, _TEXT_THEMES)},
    {"id": "style6", "name": "Confetti", "render": _r_style6,
     "ratios": ["1:1"], "slots": [],
     **_meta(False, False, 0, 0, _IMG_THEMES)},
    {"id": "style7", "name": "Dot Grid", "render": _r_style7,
     "ratios": ["1:1"], "slots": [],
     **_meta(False, False, 0, 0, _IMG_THEMES)},
]
TEMPLATE_BY_ID = {t["id"]: t for t in TEMPLATES}


# ── Recommendation engine ──────────────────────────────────────────────────────
# After upload we don't show the whole library — we filter templates compatible
# with the user's content, then randomly surface a diverse handful.

def _content_compatible(t: dict, has_title: bool, has_subtitle: bool) -> bool:
    """Whether a template can render the given combination of inputs.

    - image only (no text)  ⇒ image-only templates
    - image + title         ⇒ any text template (subtitle simply omitted)
    - image + title + sub   ⇒ text templates that support a subtitle
    (Title without an image is unsupported and never reaches here.)
    """
    is_text = bool(t.get("supportsTitle"))
    if not has_title and not has_subtitle:
        return not is_text
    if not is_text:
        return False
    if has_subtitle and not t.get("supportsSubtitle"):
        return False
    return True


def _eligible_variations(t: dict, has_title: bool, has_subtitle: bool) -> list:
    """Layout variants of one template that fit the content.

    Exact line counts are resolved by auto-fit at render time; this only
    enumerates the {title-lines × subtitle?} combinations the content permits.
    """
    if not has_title and not has_subtitle:
        return ["image"]
    variants = []
    for n in range(1, max(1, int(t.get("maxTitleLines", 1))) + 1):
        variants.append(f"{n}-line-title" + ("+subtitle" if has_subtitle else ""))
    return variants


def _pick_theme(t: dict, preferred, rng: random.Random) -> str:
    themes = t.get("themes") or ["white"]
    if preferred and preferred in themes:
        return preferred
    return rng.choice(themes)


def _as_rec(t: dict, theme: str, has_title: bool, has_subtitle: bool,
            rng: random.Random) -> dict:
    variations = _eligible_variations(t, has_title, has_subtitle)
    return {
        "template": t["id"],
        "name": t["name"],
        "theme": theme,
        "variations": variations,
        "variation": rng.choice(variations),
        "slots": t["slots"],
    }


def recommend(*, has_image: bool, has_title: bool, has_subtitle: bool,
              ratio_id: str, preferred_theme=None, count: int = 3,
              seed=None) -> list:
    """Recommend up to `count` templates for the user's content.

    Filters the library by ratio + content compatibility, then randomly picks
    options prioritising layout diversity: distinct templates first, and only
    once those are exhausted does it fall back to alternate colour themes of an
    already-picked template. This keeps repeat uploads feeling fresh while never
    padding the shortlist with near-identical colour swaps.
    """
    rng = random.Random(seed)
    if not has_image:
        return []  # title-only (no image) is unsupported

    elig = [t for t in TEMPLATES
            if ratio_id in t["ratios"]
            and _content_compatible(t, has_title, has_subtitle)]
    rng.shuffle(elig)

    recs, used = [], set()
    # pass 1 — one theme per distinct layout (maximise layout diversity)
    for t in elig:
        if len(recs) >= count:
            break
        recs.append(_as_rec(t, _pick_theme(t, preferred_theme, rng),
                            has_title, has_subtitle, rng))
        used.add(t["id"])
    # pass 2 — layouts exhausted: fill remaining slots with alternate themes
    if len(recs) < count:
        pool = []
        for t in elig:
            taken = {r["theme"] for r in recs if r["template"] == t["id"]}
            pool += [(t, th) for th in t.get("themes", []) if th not in taken]
        rng.shuffle(pool)
        for t, th in pool:
            if len(recs) >= count:
                break
            recs.append(_as_rec(t, th, has_title, has_subtitle, rng))
    return recs


# ── render + cache ─────────────────────────────────────────────────────────────

_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_CACHE_MAX = 96


def render_template(tpl_id: str, ratio_id: str, img_bytes: bytes,
                    texts: dict, out_w: int, theme: str = None) -> bytes:
    """Render a template to PNG bytes. Memoised on all inputs.

    `theme` is accepted (and keyed) so the recommendation engine's colour
    choice round-trips end-to-end; renderers are still single-palette, so it
    currently only partitions the cache until themed rendering lands.
    """
    tpl = TEMPLATE_BY_ID[tpl_id]
    ratio = RATIO_BY_ID[ratio_id]
    texts = {k: (v or "").strip() for k, v in (texts or {}).items()}
    # The title is optional in the flow: templates with a title slot fall back
    # to the brand placeholder. The subtitle stays empty — renderers skip it.
    if any(s["key"] == "title" for s in tpl["slots"]) and not texts.get("title"):
        texts["title"] = "Construction Junction"

    key = (tpl_id, ratio_id, hashlib.sha1(img_bytes).hexdigest(),
           tuple(sorted(texts.items())), out_w, theme)
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


# ── Testing templates (Figma "Style 1-4" sets, Funnnn file) ─────────────────────
# Pixel-for-pixel reproductions, each in three colour themes:
#   deep green #005618 · light green (lime) #BCF00E · white
# Design space is 270 wide (S = out_w / 270); anchors lifted straight from the
# Figma variants and expressed bottom-relative so they hold at every ratio.
#   style1  294:8047/8307/8551 — full-bleed photo, bottom gradient, corner scribble
#   style2  293:1964 &c        — theme overlay with organic blob cutout, texts top-left
#   style3  293:799 &c         — full-bleed photo, bottom panel (rounded TR), corner scribble
#   style4  266:547/282:822/883 — theme card: rounded photo, squiggle-underlined title
STYLE1_H = {"1:1": 270.0, "4:5": 337.5, "9:16": 480.0}
STYLE1_OUT = {"1:1": 1080, "4:5": 1080, "9:16": 1080}

WHITE = (255, 255, 255)
BROWN = (85, 54, 39)              # #553627 — text on white/lime themes

STYLE1_THEMES = {
    "green": {"grad": GREEN, "text": WHITE, "scribble": LIME},
    "lime":  {"grad": (186, 238, 15), "text": WHITE, "scribble": WHITE},
    "white": {"grad": WHITE, "text": BROWN, "scribble": LIME},
}
STYLE2_THEMES = {
    "green": {"bg": GREEN, "text": WHITE, "scribble": LIME},
    "lime":  {"bg": LIME,  "text": BROWN, "scribble": WHITE},
    "white": {"bg": WHITE, "text": BROWN, "scribble": LIME},
}
STYLE3_THEMES = STYLE2_THEMES     # same palette mapping (panel instead of blob)
STYLE4_THEMES = {
    "green": {"bg": GREEN, "text": WHITE, "scribble": LIME},
    "lime":  {"bg": LIME,  "text": WHITE, "scribble": WHITE},
    "white": {"bg": WHITE, "text": BROWN, "scribble": LIME},
}

_TESTING_ASSETS = os.path.join(_BASE, "static", "testing")


@functools.lru_cache(maxsize=16)
def _tasset(name: str) -> Image.Image:
    return Image.open(os.path.join(_TESTING_ASSETS, name)).convert("RGBA")


def _recolor(img: Image.Image, rgb: tuple) -> Image.Image:
    """Same alpha, solid `rgb` fill."""
    layer = Image.new("RGBA", img.size, rgb + (0,))
    layer.putalpha(img.getchannel("A"))
    return layer


def _corner_scribble(canvas, rgb, S) -> None:
    """Style-1/3 diagonal brush bleeding off the top-right corner (Figma
    "Group 59450": box right -51.63 / top -41 / 207.6×129.3, identical at
    every ratio, so it ships as one canvas-clipped 270×270 4x export)."""
    side = round(W * S)
    br = _recolor(_tasset("s1_brush.png"), rgb).resize((side, side), Image.LANCZOS)
    canvas.alpha_composite(br, (0, 0))


def _t_style1(canvas, photo, title, subtitle, H, S, theme) -> None:
    """Full-bleed photo, transparent→theme bottom gradient, bottom-anchored
    texts (subtitle top H-34; title bottom H-39 with subtitle · H-12 without;
    gradient 101/56 +26 per extra title line), corner scribble."""
    th = STYLE1_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    draw = ImageDraw.Draw(canvas)

    has_sub = bool(subtitle)
    n = len(_wrap(draw, title, _font(round(26 * S), "Bold"), 240 * S, 2))
    title_bottom = (H - 39) if has_sub else (H - 12)
    grad_h = (101 if has_sub else 56) + (26 if n == 2 else 0)

    _v_gradient(canvas, 0, H - grad_h, W, grad_h, th["grad"], 0, 255, S)
    _corner_scribble(canvas, th["scribble"], S)
    _text(canvas, title, 15, title_bottom - 26 * n, 26, 26, S, weight="Bold",
          fill=th["text"] + (255,), max_w=240, max_lines=2)
    if has_sub:
        _text(canvas, subtitle, 15, H - 34, 10, 12, S, weight="Regular",
              fill=th["text"] + (255,), max_w=240, max_lines=2)


# Style-2 arrow scribble (Figma "Group 59478"): full-canvas 4x exports per
# ratio and per subtitle state (the arrow sits 36 units higher without one).
_S2_BLOB = {"1:1": "11", "4:5": "45", "9:16": "916"}


def _t_style2(canvas, photo, title, subtitle, H, S, theme, ratio_id) -> None:
    """Full-bleed photo under a theme-coloured overlay with an organic blob
    cutout at (10,10,250,H-20); title 24/24 bold at (20,20), subtitle 10px
    six below it; rotated paint scribble bleeding off the right edge."""
    th = STYLE2_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))

    variant = "sub" if subtitle else "nosub"
    blob = _recolor(_tasset(f"s2_blob_{_S2_BLOB[ratio_id]}_{variant}.png"), th["bg"])
    bw, bh = round(250 * S), round((H - 20) * S)
    canvas.alpha_composite(blob.resize((bw, bh), Image.LANCZOS),
                           (round(10 * S), round(10 * S)))

    arrow = _recolor(_tasset(f"s2_arrow_{variant}_{_S2_BLOB[ratio_id]}.png"),
                     th["scribble"])
    canvas.alpha_composite(arrow.resize(canvas.size, Image.LANCZOS))

    ty = _text(canvas, title, 20, 20, 24, 24, S, weight="Bold",
               fill=th["text"] + (255,), max_w=230, max_lines=2)
    if subtitle:
        _text(canvas, subtitle, 20, ty + 6, 10, 12, S, weight="Regular",
              fill=th["text"] + (255,), max_w=230, max_lines=2)


def _t_style3(canvas, photo, title, subtitle, H, S, theme) -> None:
    """Full-bleed photo with a theme-coloured bottom panel (rounded top-right
    20): title top = panel top + 14, one-line subtitle top = H-27; panel height
    47+26n with subtitle · 25+26n without; Style-1 corner scribble."""
    th = STYLE3_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    draw = ImageDraw.Draw(canvas)

    has_sub = bool(subtitle)
    n = len(_wrap(draw, title, _font(round(26 * S), "Bold"), 240 * S, 2))
    panel_h = (47 if has_sub else 25) + 26 * n

    # panel: rounded_rectangle extended past the visible left/bottom edges so
    # only the top-right corner is actually rounded on canvas.
    r = 20 * S
    y0 = (H - panel_h) * S
    draw.rounded_rectangle([-r, y0, W * S - 1, H * S - 1 + r],
                           radius=r, fill=th["bg"] + (255,))

    _corner_scribble(canvas, th["scribble"], S)
    _text(canvas, title, 15, H - panel_h + 14, 26, 26, S, weight="Bold",
          fill=th["text"] + (255,), max_w=240, max_lines=2)
    if has_sub:
        _text(canvas, subtitle, 15, H - 27, 10, 12, S, weight="Regular",
              fill=th["text"] + (255,), max_w=240, max_lines=1)


def _t_style4(canvas, photo, title, subtitle, H, S, theme) -> None:
    """Theme-coloured card: rounded photo (10,10,250,·) with a soft dark bottom
    gradient, bold title whose last line sits on a squiggle underline (left 7,
    top H-61, width 161.8), subtitle at H-34. Title bottom is fixed at H-46, so
    the photo absorbs the height (photo_h = title_top - 21). The Figma set only
    defines a title+subtitle layout — without a subtitle no text is shown and
    the photo fills the card."""
    th = STYLE4_THEMES[theme]
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W * S, H * S], fill=th["bg"] + (255,))

    if not subtitle:
        iw, ih = round(250 * S), round((H - 20) * S)
        canvas.paste(_cover(photo, iw, ih), (round(10 * S), round(10 * S)),
                     _rounded_mask(iw, ih, round(5 * S)))
        return

    n = len(_wrap(draw, title, _font(round(26 * S), "Bold"), 240 * S, 2))
    title_top = H - 46 - 26 * n
    photo_h = title_top - 21

    iw, ih = round(250 * S), round(photo_h * S)
    img = _cover(photo, iw, ih)
    canvas.paste(img, (round(10 * S), round(10 * S)),
                 _rounded_mask(iw, ih, round(5 * S)))
    _v_gradient(canvas, 10, 10 + photo_h - 53, 250, 53, (0, 0, 0), 0, 178, S,
                radius=5)

    squig = _recolor(_asset("proc_s2_scribble.png"), th["scribble"])
    sw = round(161.8 * S)
    sh = round(squig.height * sw / squig.width)
    canvas.alpha_composite(squig.resize((sw, sh), Image.LANCZOS),
                           (round(7 * S), round((H - 61) * S)))

    _text(canvas, title, 15, title_top, 26, 26, S, weight="Bold",
          fill=th["text"] + (255,), max_w=240, max_lines=2)
    _text(canvas, subtitle, 15, H - 34, 10, 12, S, weight="Regular",
          fill=th["text"] + (255,), max_w=240, max_lines=2)


# Style 5 — mirror of Style 4: title + squiggle underline at the TOP, rounded
# photo filling the bottom (bottom edge fixed at H-10, no dark gradient).
STYLE5_THEMES = {
    "green": {"bg": GREEN, "text": WHITE, "scribble": LIME},
    "lime":  {"bg": LIME,  "text": BROWN, "scribble": WHITE},
    "white": {"bg": WHITE, "text": BROWN, "scribble": LIME},
}


def _t_style5(canvas, photo, title, subtitle, H, S, theme) -> None:
    th = STYLE5_THEMES[theme]
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W * S, H * S], fill=th["bg"] + (255,))

    n = len(_wrap(draw, title, _font(round(26 * S), "Bold"), 240 * S, 2))
    photo_top = 15 + 26 * n + (56 if subtitle else 37)
    photo_h = (H - 10) - photo_top

    iw, ih = round(250 * S), round(photo_h * S)
    canvas.paste(_cover(photo, iw, ih), (round(10 * S), round(photo_top * S)),
                 _rounded_mask(iw, ih, round(4 * S)))

    # squiggle underlining the last title line (same asset as Style 4)
    squig = _recolor(_asset("proc_s2_scribble.png"), th["scribble"])
    sw = round(161.8 * S)
    sh = round(squig.height * sw / squig.width)
    canvas.alpha_composite(squig.resize((sw, sh), Image.LANCZOS),
                           (round(7 * S), round((15 + 26 * (n - 1) + 12) * S)))

    _text(canvas, title, 15, 15, 26, 26, S, weight="Bold",
          fill=th["text"] + (255,), max_w=240, max_lines=2)
    if subtitle:
        _text(canvas, subtitle, 15, 15 + 26 * n + 18, 10, 12, S, weight="Regular",
              fill=th["text"] + (255,), max_w=240, max_lines=2)


# Style 6 — full-bleed photo, a theme-coloured gradient banner across the top
# (opaque→transparent) carrying the title/subtitle, plus a scribble bleeding off
# the bottom-left.
STYLE6_THEMES = {
    "green": {"grad": GREEN, "text": WHITE, "scribble": LIME},
    "lime":  {"grad": LIME,  "text": BROWN, "scribble": WHITE},
    "white": {"grad": WHITE, "text": BROWN, "scribble": LIME},
}


def _top_gradient(canvas, h_units, rgb, S) -> None:
    """Vertical banner: solid `rgb` at the top → transparent at the bottom
    (Figma stops: 1.0 → 0.58@79.3% → 0)."""
    gh, gw = round(h_units * S), round(W * S)
    ramp = []
    for i in range(gh):
        t = i / max(1, gh - 1)
        a = 255 + (148 - 255) * (t / 0.793269) if t <= 0.793269 \
            else 148 + (0 - 148) * ((t - 0.793269) / (1 - 0.793269))
        ramp.append(int(a))
    col = Image.new("L", (1, gh)); col.putdata(ramp)
    layer = Image.new("RGBA", (gw, gh), rgb + (255,))
    layer.putalpha(col.resize((gw, gh)))
    canvas.alpha_composite(layer, (0, 0))


# Banner heights per ratio (Figma "Rectangle 5", with / without subtitle) and
# the bottom-left arrow scribble ("Group 59478", full-canvas 4x export, drawn
# under the banner; identical with or without subtitle).
_S6_BANNER = {"1:1": (95, 72), "4:5": (96, 69), "9:16": (99, 63)}
_S6_TAG = {"1:1": "11", "4:5": "45", "9:16": "916"}


def _t_style6(canvas, photo, title, subtitle, H, S, theme, ratio_id) -> None:
    th = STYLE6_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))

    arrow = _recolor(_tasset(f"s6_arrow_{_S6_TAG[ratio_id]}.png"), th["scribble"])
    canvas.alpha_composite(arrow.resize(canvas.size, Image.LANCZOS))

    hs, hn = _S6_BANNER[ratio_id]
    _top_gradient(canvas, hs if subtitle else hn, th["grad"], S)

    _text(canvas, title, 15, 15, 26, 26, S, weight="Bold",
          fill=th["text"] + (255,), max_w=240, max_lines=2)
    if subtitle:
        _text(canvas, subtitle, 15, 46, 10, 12, S, weight="Regular",
              fill=th["text"] + (255,), max_w=240, max_lines=2)


# Style 7 — full-bleed photo under an organic theme-coloured frame (Figma
# "Subtract", distinct per ratio AND per subtitle state), small scribble lines
# right of the title ("Group 59432"), texts bottom-left. With a subtitle the
# title is top-anchored at H-84 and the one-line subtitle sits at H-30;
# without one the title block is bottom-anchored at H-16. Both decorations
# ship as full-canvas 4x exports.
STYLE7_THEMES = {
    "green": {"frame": GREEN, "text": WHITE, "scribble": LIME},
    "lime":  {"frame": LIME,  "text": BROWN, "scribble": WHITE},
    "white": {"frame": WHITE, "text": BROWN, "scribble": LIME},
}
_S7_TAG = {"1:1": "11", "4:5": "45", "9:16": "916"}


def _t_style7(canvas, photo, title, subtitle, H, S, theme, ratio_id) -> None:
    th = STYLE7_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))

    tag, variant = _S7_TAG[ratio_id], "sub" if subtitle else "nosub"
    frame = _recolor(_tasset(f"s7_frame_{tag}_{variant}.png"), th["frame"])
    canvas.alpha_composite(frame.resize(canvas.size, Image.LANCZOS))

    if subtitle:
        _text(canvas, title, 20, H - 84, 24, 26, S, weight="Bold",
              fill=th["text"] + (255,), max_w=230, max_lines=2)
        _text(canvas, subtitle, 20, H - 30, 10, 12, S, weight="Regular",
              fill=th["text"] + (255,), max_w=230, max_lines=1)
    else:
        n = len(_wrap(ImageDraw.Draw(canvas), title,
                      _font(round(24 * S), "Bold"), 230 * S, 2))
        _text(canvas, title, 20, H - 16 - 26 * n, 24, 26, S, weight="Bold",
              fill=th["text"] + (255,), max_w=230, max_lines=2)

    # scribble sits above the text in Figma's z-order
    scrib = _recolor(_tasset(f"s7_scrib_{tag}_{variant}.png"), th["scribble"])
    canvas.alpha_composite(scrib.resize(canvas.size, Image.LANCZOS))


# Style 8 — ornate plaque-masked photo over corner accents, an arrow scribble
# off the top-left, right-aligned bold title. With a subtitle the one-line
# muted description sits right under the title; without one a brush squiggle
# underlines the title (baked into the "over" layer). Rebuilt from Figma
# 285:3455 / 294:5467 / 294:5548 (white / lime / deep-green sets); the accents,
# plaque mask and arrow(+squiggle) ship as full-canvas 4x layer exports.
STYLE8_THEMES = {
    "white": {"bg": WHITE, "acc": GREEN, "deco": LIME,
              "title": GREEN, "sub": (0, 0, 0, 153)},
    "lime":  {"bg": LIME,  "acc": GREEN, "deco": WHITE,
              "title": GREEN, "sub": (0, 0, 0, 153)},
    "green": {"bg": GREEN, "acc": WHITE, "deco": LIME,
              "title": WHITE, "sub": (255, 255, 255, 153)},
}
_S8_TAG = {"1:1": "11", "4:5": "45", "9:16": "916"}
_S8_SUB_TOP = {"1:1": 18, "4:5": 20, "9:16": 20}   # subtitle top = H - this


def _t_style8(canvas, photo, title, subtitle, H, S, theme, ratio_id) -> None:
    th = STYLE8_THEMES[theme]
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W * S, H * S], fill=th["bg"] + (255,))
    tag = _S8_TAG[ratio_id]

    acc = _recolor(_tasset(f"s8_acc_{tag}.png"), th["acc"])
    canvas.alpha_composite(acc.resize(canvas.size, Image.LANCZOS))

    mask = _tasset(f"s8_mask_{tag}.png").resize(canvas.size, Image.LANCZOS) \
        .getchannel("A")
    bb = mask.getbbox()
    canvas.paste(_cover(photo, bb[2] - bb[0], bb[3] - bb[1]),
                 (bb[0], bb[1]), mask.crop(bb))

    variant = "desc" if subtitle else "nodesc"
    over = _recolor(_tasset(f"s8_over_{tag}_{variant}.png"), th["deco"])
    canvas.alpha_composite(over.resize(canvas.size, Image.LANCZOS))

    n = len(_wrap(draw, title, _font(round(26 * S), "Bold"), 240 * S, 2))
    if subtitle:
        sub_top = H - _S8_SUB_TOP[ratio_id]
        _text(canvas, title, 15, sub_top - 26 * n, 26, 26, S, weight="Bold",
              fill=th["title"] + (255,), max_w=240, max_lines=2, align="right")
        _text(canvas, subtitle, 15, sub_top, 10, 12, S, weight="Regular",
              fill=th["sub"], max_w=240, max_lines=1, align="right")
    else:
        _text(canvas, title, 15, H - 22 - 26 * n, 26, 26, S, weight="Bold",
              fill=th["title"] + (255,), max_w=240, max_lines=2, align="right")


# Style 9 — full-bleed photo under a single fixed confetti overlay (one design,
# no theme, no text). The overlay PNG is placed per ratio (Figma offsets).
_S9_OVERLAY = {
    "1:1":  ("11",  272, 478, -1, -104),
    "4:5":  ("45",  272, 478, -1, -43),
    "9:16": ("916", 278, 489, -4, -5),
}


def _t_style9(canvas, photo, title, subtitle, H, S, theme) -> None:
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    tag, ow, oh, ox, oy = _S9_OVERLAY[["1:1", "4:5", "9:16"][
        [270.0, 337.5, 480.0].index(H)]]
    ov = _tasset(f"s9_overlay_{tag}.png").resize((round(ow * S), round(oh * S)), Image.LANCZOS)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(ov, (round(ox * S), round(oy * S)))
    canvas.alpha_composite(layer)


# ── Secondary styles (Figma "Secondary Style 1-3" sets, Funnnn file) ────────────
# Shared core: full-bleed photo, 40% black overlay, centred 26px title
# (px-36 → text box 198 wide, centred both axes); the designs carry no subtitle,
# so it is intentionally not rendered. Decoration sits on top of the text.
#   style10  311:1044/1054, 313:555 — Secondary 1: solid band frame from canvas
#            edge to an inset-10 cutout, TL/BR corners r10, TR/BL square
#            (drawn from the Figma SVG path; SemiBold title, leading 1.2)
#   style11  294:5636/5630/5642, 315:734/738/742 — Secondary 2: hand-drawn
#            double-line frame (4x SVG export s11_frame_*.png; SemiBold, 1.2)
#   style12  276:609/611/607, 317:810/882/954 — Secondary 3 scribble layout 1:
#            brush stroke off the top-right + zigzag off the bottom-left
#            (s12_scrib_*.png; Bold title, leading 26)
#   style13  276:608/610/606, 317:818/890/970 — Secondary 3 scribble layout 2:
#            three short strokes top-right + curved arrow bottom-left
#            (s13_scrib_*.png; Bold title, leading 26)
# Figma ships lime + white variants (style10 also a title-less deep-green one);
# the green theme recolours the decoration #005618 and keeps white text.
_SEC_TAG = {"1:1": "11", "4:5": "45", "9:16": "916"}
_SEC_DECO = {"green": GREEN, "lime": LIME, "white": WHITE}


def _sec_core(canvas, photo, title, H, S, weight, lh, theme) -> None:
    """Photo + 40% black overlay + title centred in the canvas (max 2 lines).
    Lime theme titles are lime (per Figma); white/green themes use white."""
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    if not title:
        return
    canvas.alpha_composite(Image.new("RGBA", canvas.size, (0, 0, 0, 102)))
    draw = ImageDraw.Draw(canvas)
    n = len(_wrap(draw, title, _font(round(26 * S), weight), 198 * S, 2))
    fill = (LIME if theme == "lime" else WHITE) + (255,)
    _text(canvas, title, 36, (H - lh * n) / 2, 26, lh, S, weight=weight,
          fill=fill, max_w=198, max_lines=2, align="center")


def _sec_scribble(canvas, name: str, rgb: tuple) -> None:
    """Full-canvas 4x decoration PNG, recoloured and fitted to the canvas."""
    deco = _tasset(name).resize(canvas.size, Image.LANCZOS)
    canvas.alpha_composite(_recolor(deco, rgb))


def _t_style10(canvas, photo, title, subtitle, H, S, theme) -> None:
    """Secondary 1 — solid band frame: canvas edge down to a rounded cutout at
    inset 10 (radius 10 on the TL/BR corners only, Figma-exact path)."""
    _sec_core(canvas, photo, title, H, S, "SemiBold", 31.2, theme)
    mask = Image.new("L", canvas.size, 255)
    ImageDraw.Draw(mask).rounded_rectangle(
        [round(10 * S), round(10 * S),
         round((W - 10) * S) - 1, round((H - 10) * S) - 1],
        radius=round(10 * S), fill=0, corners=(True, False, True, False))
    band = Image.new("RGBA", canvas.size, _SEC_DECO[theme] + (255,))
    band.putalpha(mask)
    canvas.alpha_composite(band)


def _t_style11(canvas, photo, title, subtitle, H, S, theme, ratio_id) -> None:
    """Secondary 2 — hand-drawn double-line frame over the centred title."""
    _sec_core(canvas, photo, title, H, S, "SemiBold", 31.2, theme)
    _sec_scribble(canvas, f"s11_frame_{_SEC_TAG[ratio_id]}.png", _SEC_DECO[theme])


def _t_style12(canvas, photo, title, subtitle, H, S, theme, ratio_id) -> None:
    """Secondary 3, scribble layout 1 — top-right brush + bottom-left zigzag."""
    _sec_core(canvas, photo, title, H, S, "Bold", 26, theme)
    _sec_scribble(canvas, f"s12_scrib_{_SEC_TAG[ratio_id]}.png", _SEC_DECO[theme])


def _t_style13(canvas, photo, title, subtitle, H, S, theme, ratio_id) -> None:
    """Secondary 3, scribble layout 2 — short strokes + curved arrow."""
    _sec_core(canvas, photo, title, H, S, "Bold", 26, theme)
    _sec_scribble(canvas, f"s13_scrib_{_SEC_TAG[ratio_id]}.png", _SEC_DECO[theme])


TESTING_STYLES = {
    "style1": _t_style1,
    "style2": _t_style2,
    "style3": _t_style3,
    "style4": _t_style4,
    "style5": _t_style5,
    "style6": _t_style6,
    "style7": _t_style7,
    "style8": _t_style8,
    "style9": _t_style9,
    "style10": _t_style10,
    "style11": _t_style11,
    "style12": _t_style12,
    "style13": _t_style13,
}
# Styles whose renderer signature also takes ratio_id (ratio-specific assets).
_STYLES_NEED_RATIO = {"style2", "style6", "style7", "style8",
                      "style11", "style12", "style13"}
TESTING_THEMES = ["green", "lime", "white"]


def render_testing(style: str, theme: str, ratio_id: str, img_bytes: bytes,
                   title: str, subtitle: str, out_w: int) -> bytes:
    """Render one Testing style+theme to PNG bytes. Memoised on all inputs.
    An empty title falls back to the brand name; empty subtitle is omitted."""
    title = (title or "").strip() or "Construction Junction"
    subtitle = (subtitle or "").strip()
    key = ("testing", style, theme, ratio_id,
           hashlib.sha1(img_bytes).hexdigest(), title, subtitle, out_w)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    H = STYLE1_H[ratio_id]
    S = out_w / W
    photo = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    canvas = Image.new("RGBA", (out_w, round(H * S)), (255, 255, 255, 255))
    fn = TESTING_STYLES[style]
    if style in _STYLES_NEED_RATIO:            # needs the ratio for its assets
        fn(canvas, photo, title, subtitle, H, S, theme, ratio_id)
    else:
        fn(canvas, photo, title, subtitle, H, S, theme)
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    png = buf.getvalue()

    _CACHE[key] = png
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return png
