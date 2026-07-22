"""Social template engine — framework-free, Figma-exact.

Templates replicate the CJ Studio Figma file (Funnnn) as declarative layouts in
a 270-wide design space. Decoration assets are 4x Figma exports processed into
white+alpha masks in static/social2/ (recoloured per theme at render time);
see the asset notes at the bottom of this docstring.

    render_testing(style, theme, ratio_id, img_bytes, title, subtitle, out_w)
        -> PNG bytes   (name kept for API compatibility with main.py)

Template library (matches the Figma component sets 1:1):
    cover1..cover7   Cover    — image + title + optional subtitle, 3 themes
    textonly         Text     — title + optional subtitle, NO image, 3 themes
    sec1a..sec1d     Secondary 1 styles 1-4 — image + title + optional
                     subtitle + fixed byline, single colour scheme
    sec2/sec3        Secondary 2/3 — image + title (no subtitle)
    sec4a/sec4b      Secondary 4 scribble styles 1/2 — image + title
    imageonly        Image    — photo + confetti overlay, no text

Colour variants: the pickers map green→Figma "Dark", lime→"Light",
white→"White". Sets that ship without a green/dark variant (sec2..sec4,
imageonly) synthesise it by recolouring the decoration — same precedent as the
previous engine generation.

Typography: IBM Plex Serif Bold Italic (titles) / Regular (subtitles),
IBM Plex Sans Medium (Secondary-1 byline; stands in for Helvetica Neue
Medium, which is not redistributable).

TEMPLATES carries per-template metadata (category, requiresImage,
supportsTitle/Subtitle) so the frontend can sort/dim templates by what the
user's content supports.
"""
import functools
import hashlib
import io
import os
import re
from collections import OrderedDict

from PIL import Image, ImageDraw, ImageFont, ImageOps

GREEN = (0, 86, 24)            # CJ brand green  #005618
LIME = (188, 240, 14)          # accent          #BCF00E
WHITE = (255, 255, 255)
BROWN = (85, 54, 39)           # serif text      #553627

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS = os.path.join(_BASE, "static", "social2")
_FONTS = os.path.join(_BASE, "fonts")

W = 270                        # design-space width, always
H_BY_RATIO = {"1:1": 270.0, "4:5": 337.5, "9:16": 480.0}
STYLE1_H = H_BY_RATIO          # legacy alias used by main.py
_TAG = {"1:1": "11", "4:5": "45", "9:16": "916"}


@functools.lru_cache(maxsize=64)
def _asset(name: str) -> Image.Image:
    return Image.open(os.path.join(_ASSETS, name)).convert("RGBA")


@functools.lru_cache(maxsize=64)
def _font(kind: str, size_px: int) -> ImageFont.FreeTypeFont:
    path = {
        "title": "IBMPlexSerif-BoldItalic.ttf",
        "body": "IBMPlexSerif-Regular.ttf",
        "sans": "IBMPlexSans-Medium.ttf",
    }[kind]
    f = ImageFont.truetype(os.path.join(_FONTS, path), size_px)
    if kind == "sans":
        try:
            f.set_variation_by_name("Medium")
        except Exception:
            pass
    return f


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    return ImageOps.fit(img, (max(1, w), max(1, h)), Image.LANCZOS)


def _recolor(img: Image.Image, rgb: tuple, opacity: float = 1.0) -> Image.Image:
    layer = Image.new("RGBA", img.size, rgb + (0,))
    a = img.getchannel("A")
    if opacity < 1.0:
        a = a.point(lambda v: int(v * opacity))
    layer.putalpha(a)
    return layer


def _deco(canvas, name, rgb, x, y, w, h, S, opacity: float = 1.0) -> None:
    """Recoloured decoration stretched to (w,h) design units at (x,y)."""
    a = _recolor(_asset(name), rgb, opacity)
    a = a.resize((max(1, round(w * S)), max(1, round(h * S))), Image.LANCZOS)
    canvas.alpha_composite(a, (round(x * S), round(y * S)))


def _capitalize(text: str) -> str:
    """CSS text-transform: capitalize (first letter of each word only)."""
    return " ".join(w[:1].upper() + w[1:] if w else w for w in text.split(" "))


# Emoji and pictographs: the template fonts have no glyphs for them (they
# render as tofu boxes), so they are silently stripped before rendering.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"      # pictographs, emoticons, symbols
    "\U0001F1E6-\U0001F1FF"      # regional indicators (flags)
    "☀-➿"              # misc symbols + dingbats
    "⬀-⯿"              # stars, misc arrows
    "︀-️"              # variation selectors
    "‍"                     # zero-width joiner
    "]+"
)


def _strip_emoji(text: str) -> str:
    return " ".join(_EMOJI_RE.sub("", text).split())


# Set whenever a title is shortened (ellipsised past max_lines); reset per
# render and surfaced to the frontend as an X-Title-Truncated header so the
# editor can hint that part of the title won't fit the chosen template.
_TRUNCATED = False


def _fit_px(draw, text, size_px, kind, max_w_px) -> int:
    """Shrink-to-fit: largest font size ≤ size_px whose longest word fits
    max_w_px, floored at 60% — overlong words are char-broken by _wrap after
    the floor. Returns size_px unchanged for ordinary text."""
    words = text.split()
    if not words:
        return size_px
    font = _font(kind, size_px)
    longest = max(words, key=lambda w: draw.textlength(w, font=font))
    if draw.textlength(longest, font=font) <= max_w_px:
        return size_px
    floor = max(8, int(size_px * 0.6))
    for s in range(size_px - 1, floor, -1):
        if draw.textlength(longest, font=_font(kind, s)) <= max_w_px:
            return s
    return floor


def _split_word(draw, word, font, max_w) -> list:
    parts, cur = [], ""
    for ch in word:
        if not cur or draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            parts.append(cur)
            cur = ch
    return parts + ([cur] if cur else [])


def _wrap(draw, text, font, max_w, max_lines, track=False) -> list:
    global _TRUNCATED
    words = []
    for w in text.split():
        if draw.textlength(w, font=font) > max_w:      # last resort: char-break
            words += _split_word(draw, w, font, max_w)
        else:
            words.append(w)
    lines, cur = [], ""
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
        if track:
            _TRUNCATED = True
    return lines


def _text(canvas, text, x, y, size, lh, S, kind="title", fill=(255, 255, 255, 255),
          max_w=240, max_lines=2, align="left") -> float:
    """Draw wrapped text; returns the y just below the last line (units)."""
    if not text:
        return y
    draw = ImageDraw.Draw(canvas)
    px = _fit_px(draw, text, round(size * S), kind, max_w * S)
    font = _font(kind, px)
    lines = _wrap(draw, text, font, max_w * S, max_lines, track=(kind == "title"))
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


def _n_lines(canvas, text, size, S, max_w, max_lines, kind="title") -> int:
    draw = ImageDraw.Draw(canvas)
    px = _fit_px(draw, text, round(size * S), kind, max_w * S)
    return len(_wrap(draw, text, _font(kind, px), max_w * S, max_lines))


def _v_gradient(canvas, x, y, w, h, rgb, a0, a1, S) -> None:
    gw, gh = max(1, round(w * S)), max(1, round(h * S))
    grad = Image.new("L", (1, gh))
    grad.putdata([int(a0 + (a1 - a0) * i / max(1, gh - 1)) for i in range(gh)])
    layer = Image.new("RGBA", (gw, gh), rgb + (255,))
    layer.putalpha(grad.resize((gw, gh)))
    canvas.alpha_composite(layer, (round(x * S), round(y * S)))


SUB_LH = 11.7                  # 9px IBM Plex Serif, Figma leading "normal"


# ── Cover 1 / 4 / 7 — theme card, text block + photo ──────────────────────────
# Shared geometry (Figma autolayout): content inset 15/10/10, gap 8 between
# text block (min-h 48, px 5 → x15 w240) and image (x10 w250). Title 26/26
# Bold Italic capitalize (≤2 lines), 18 gap, subtitle 9 (≤2 lines). A lime/
# white squiggle underlines the last title line (Stroke group: left -11,
# h 37.6, bottom = block bottom + 20.6 → top = last line top + 26 - 17.2).
CARD_THEMES = {
    "green": {"bg": GREEN, "text": WHITE, "squiggle": LIME, "plaque": WHITE},
    "lime":  {"bg": LIME, "text": BROWN, "squiggle": WHITE, "plaque": BROWN},
    "white": {"bg": WHITE, "text": BROWN, "squiggle": LIME, "plaque": LIME},
}
SQUIG_W, SQUIG_H = 161.8, 37.75


def _card_text_h(n, m, has_sub):
    """Height of the card text block. Figma's min-h 48 is one title line (26)
    plus room for the squiggle's 20.6-unit overhang below it; with a subtitle
    the subtitle block (18 gap + lines) provides that room, without one we
    keep the 22-unit allowance so the stroke never gets covered or clipped —
    that's the responsive intent behind the fixed 48."""
    return max(48.0, n * 26 + (18 + m * SUB_LH if has_sub else 22))


def _card_text(canvas, title, subtitle, top, th, S, align="left") -> None:
    n = _n_lines(canvas, title, 26, S, 240, 2)
    sx = 113 if align == "right" else 4
    _deco(canvas, "squiggle.png", th["squiggle"],
          sx, top + 26 * n - 17.2, SQUIG_W, SQUIG_H, S)
    _text(canvas, title, 15, top, 26, 26, S, fill=th["text"] + (255,),
          max_w=240, max_lines=2, align=align)
    if subtitle:
        _text(canvas, subtitle, 15, top + 26 * n + 18, 9, SUB_LH, S, kind="body",
              fill=th["text"] + (255,), max_w=240, max_lines=2, align=align)


def _r_cover1(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    th = CARD_THEMES[theme]
    ImageDraw.Draw(canvas).rectangle([0, 0, W * S, H * S], fill=th["bg"] + (255,))
    n = _n_lines(canvas, title, 26, S, 240, 2)
    m = _n_lines(canvas, subtitle, 9, S, 240, 2, "body") if subtitle else 0
    text_h = _card_text_h(n, m, bool(subtitle))
    _card_text(canvas, title, subtitle, 15, th, S)
    iw, ih = round(250 * S), round((H - 10 - (15 + text_h + 8)) * S)
    canvas.paste(_cover(photo, iw, ih), (round(10 * S), round((15 + text_h + 8) * S)))


def _r_cover4(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    th = CARD_THEMES[theme]
    ImageDraw.Draw(canvas).rectangle([0, 0, W * S, H * S], fill=th["bg"] + (255,))
    n = _n_lines(canvas, title, 26, S, 240, 2)
    m = _n_lines(canvas, subtitle, 9, S, 240, 2, "body") if subtitle else 0
    text_h = _card_text_h(n, m, bool(subtitle))
    text_top = H - 10 - text_h
    iw, ih = round(250 * S), round((text_top - 8 - 15) * S)
    canvas.paste(_cover(photo, iw, ih), (round(10 * S), round(15 * S)))
    _card_text(canvas, title, subtitle, text_top, th, S)


def _r_cover7(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    """Cover 4 layout, right-aligned text; photo in an ornate plaque mask with
    an offset theme-coloured back plaque (offset ≈ 4.97% of plaque height)."""
    th = CARD_THEMES[theme]
    ImageDraw.Draw(canvas).rectangle([0, 0, W * S, H * S], fill=th["bg"] + (255,))
    n = _n_lines(canvas, title, 26, S, 240, 2)
    m = _n_lines(canvas, subtitle, 9, S, 240, 2, "body") if subtitle else 0
    text_h = _card_text_h(n, m, bool(subtitle))
    text_top = H - 10 - text_h

    area_h = text_top - 8 - 15
    pl_h = area_h / 1.0497
    off = area_h - pl_h
    plaque = _asset(f"c7_plaque_{_TAG[ratio_id]}.png")
    pw, ph = round(240.4 * S), round(pl_h * S)
    mask = plaque.getchannel("A").resize((pw, ph), Image.LANCZOS)
    _deco(canvas, f"c7_plaque_{_TAG[ratio_id]}.png", th["plaque"],
          10, 15, 240.4, pl_h, S)
    canvas.paste(_cover(photo, pw, ph),
                 (round(19.6 * S), round((15 + off) * S)), mask)
    _card_text(canvas, title, subtitle, text_top, th, S, align="right")


# ── Cover 3 / 6 — full-bleed photo + gradient text band ───────────────────────
GRAD_THEMES = {
    "green": {"fill": GREEN, "text": WHITE, "brush": LIME},
    "lime":  {"fill": LIME, "text": WHITE, "brush": WHITE},
    "white": {"fill": WHITE, "text": BROWN, "brush": LIME},
}


def _r_cover3(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    """Bottom band: pt 30 / px 15 / pb 15, title 26 + 6 + subtitle 9; gradient
    transparent→theme behind the band; corner brush off the top-right (same
    box as the legacy Style-1 brush: full-canvas 270×270 export)."""
    th = GRAD_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    n = _n_lines(canvas, title, 26, S, 240, 2)
    m = _n_lines(canvas, subtitle, 9, S, 240, 2, "body") if subtitle else 0
    band_h = 30 + 26 * n + (6 + m * SUB_LH if subtitle else 0) + 15
    _v_gradient(canvas, 0, H - band_h, W, band_h, th["fill"], 0, 255, S)
    side = round(W * S)
    brush = _recolor(_asset("corner_brush.png"), th["brush"]).resize((side, side), Image.LANCZOS)
    canvas.alpha_composite(brush, (0, 0))
    ty = _text(canvas, _capitalize(title), 15, H - band_h + 30, 26, 26, S,
               fill=th["text"] + (255,), max_w=240, max_lines=2)
    if subtitle:
        _text(canvas, subtitle, 15, ty + 6, 9, SUB_LH, S, kind="body",
              fill=th["text"] + (255,), max_w=240, max_lines=2)


def _r_cover6(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    """Top band: pt 15 / px 15 / pb 40, gradient theme→transparent."""
    th = GRAD_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    n = _n_lines(canvas, title, 26, S, 240, 2)
    m = _n_lines(canvas, subtitle, 9, S, 240, 2, "body") if subtitle else 0
    band_h = 15 + 26 * n + (6 + m * SUB_LH if subtitle else 0) + 40
    _v_gradient(canvas, 0, 0, W, band_h, th["fill"], 255, 0, S)
    ty = _text(canvas, _capitalize(title), 15, 15, 26, 26, S,
               fill=th["text"] + (255,), max_w=240, max_lines=2)
    if subtitle:
        _text(canvas, subtitle, 15, ty + 6, 9, SUB_LH, S, kind="body",
              fill=th["text"] + (255,), max_w=240, max_lines=2)


# ── Cover 5 — bottom panel (rounded top-right) + claw scribble ────────────────
PANEL_THEMES = {
    "green": {"panel": GREEN, "text": WHITE, "claw": LIME},
    "lime":  {"panel": LIME, "text": BROWN, "claw": WHITE},
    "white": {"panel": WHITE, "text": BROWN, "claw": LIME},
}


def _r_cover5(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    th = PANEL_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    n = _n_lines(canvas, title, 26, S, 240, 2)
    panel_h = 10 + 26 * n + (6 + SUB_LH if subtitle else 0) + 15
    r = 20 * S
    ImageDraw.Draw(canvas).rounded_rectangle(
        [-r, (H - panel_h) * S, W * S - 1, H * S - 1 + r],
        radius=r, fill=th["panel"] + (255,),
        corners=(False, True, False, False))
    _deco(canvas, "c5_claw.png", th["claw"], 191, 5, 73.25, 81, S)
    ty = _text(canvas, _capitalize(title), 15, H - panel_h + 10, 26, 26, S,
               fill=th["text"] + (255,), max_w=240, max_lines=2)
    if subtitle:
        _text(canvas, subtitle, 15, ty + 6, 9, SUB_LH, S, kind="body",
              fill=th["text"] + (255,), max_w=240, max_lines=1)


# ── Cover 2 — organic blob overlay with photo cutout ──────────────────────────
# Blob overlay fills (10,10,250,H-20); shape differs per ratio × subtitle ×
# title-line-count (12 Figma exports). Title (20,20) w230 capitalize; subtitle
# top 52 (+26 per extra title line); lime/white arrow at top-left, 20 lower
# for two-line titles.
COVER2_THEMES = {
    "green": {"blob": GREEN, "title": WHITE, "sub": (255, 255, 255, 204), "arrow": LIME},
    "lime":  {"blob": LIME, "title": BROWN, "sub": BROWN + (153,), "arrow": WHITE},
    "white": {"blob": WHITE, "title": BROWN, "sub": BROWN + (255,), "arrow": LIME},
}


def _r_cover2(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    th = COVER2_THEMES[theme]
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    n = _n_lines(canvas, title, 26, S, 230, 2)
    blob = f"c2_blob_{_TAG[ratio_id]}_{'sub' if subtitle else 'nosub'}_{n}.png"
    _deco(canvas, blob, th["blob"], 10, 10, 250, H - 20, S)
    _text(canvas, _capitalize(title), 20, 20, 26, 26, S,
          fill=th["title"] + (255,), max_w=230, max_lines=2)
    if subtitle:
        _text(canvas, subtitle, 20, 52 + 26 * (n - 1), 9, SUB_LH, S, kind="body",
              fill=th["sub"], max_w=230, max_lines=2)
    _deco(canvas, "c2_arrow.png", th["arrow"],
          -0.5, 82 + 20 * (n - 1), 101.5, 57.25, S)


# ── Text Only — theme card, logo + centred text, double border ────────────────
TEXTONLY_THEMES = {
    "green": {"bg": GREEN, "border": LIME, "text": WHITE, "logo": WHITE},
    "lime":  {"bg": LIME, "border": GREEN, "text": BROWN, "logo": BROWN},
    "white": {"bg": WHITE, "border": LIME, "text": BROWN, "logo": BROWN},
}


def _r_textonly(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    th = TEXTONLY_THEMES[theme]
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W * S, H * S], fill=th["bg"] + (255,))
    draw.rectangle([round(10 * S), round(10 * S),
                    round((W - 10) * S) - 1, round((H - 10) * S) - 1],
                   outline=th["border"] + (178,), width=max(1, round(3 * S)))

    n = _n_lines(canvas, title, 26, S, 198, 2)
    m = _n_lines(canvas, subtitle, 9, S, 153, 3, "body") if subtitle else 0
    total = 26 + 44 + 31.2 * n + (32 + 1.5 + 6 + m * SUB_LH if subtitle else 0)
    top = H / 2 - 8.5 - total / 2

    _deco(canvas, "cj_logo.png", th["logo"], (W - 62.4) / 2, top, 62.4, 26, S)
    _text(canvas, title, 36, top + 26 + 44, 26, 31.2, S,
          fill=th["text"] + (255,), max_w=198, max_lines=2, align="center")
    if subtitle:
        by = top + 26 + 44 + 31.2 * n + 32
        bx = (W - 153) / 2
        draw.rectangle([round(bx * S), round(by * S),
                        round((bx + 153) * S), round((by + 1.2) * S)],
                       fill=th["text"] + (255,))
        _text(canvas, subtitle, bx, by + 1.5 + 6, 9, SUB_LH, S, kind="body",
              fill=th["text"] + (255,), max_w=153, max_lines=3)


# ── Secondary 1 (styles 1-4) — photo, 24% shade, border, logo + centred text ──
# Single colour scheme: white text/border/logo, lime scribbles. Styles differ
# only in scribble art (and style 4 anchors the text bottom-left instead).
BYLINE = "By Construction Junction"


def _sec1_base(canvas, photo, H, S) -> ImageDraw.ImageDraw:
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    canvas.alpha_composite(Image.new("RGBA", canvas.size, (0, 0, 0, 61)))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([round(10 * S), round(10 * S),
                    round((W - 10) * S) - 1, round((H - 10) * S) - 1],
                   outline=(255, 255, 255, 178), width=max(1, round(3 * S)))
    return draw


def _sec1_centre(canvas, title, subtitle, H, S, block_w) -> None:
    draw = ImageDraw.Draw(canvas)
    n = _n_lines(canvas, title, 26, S, 198, 2)
    m = _n_lines(canvas, subtitle, 10, S, block_w, 2, "body") if subtitle else 0
    total = 26 + 44 + 31.2 * n + 32 + 1.5 + 6 + (m * 12 + 6 if subtitle else 0) + 8.4
    top = H / 2 + 0.5 - total / 2
    _deco(canvas, "cj_logo.png", WHITE, (W - 62.4) / 2, top, 62.4, 26, S)
    _text(canvas, title, 36, top + 26 + 44, 26, 31.2, S,
          fill=(255, 255, 255, 255), max_w=198, max_lines=2, align="center")
    by = top + 26 + 44 + 31.2 * n + 32
    bx = (W - block_w) / 2
    draw.rectangle([round(bx * S), round(by * S),
                    round((bx + block_w) * S), round((by + 1.2) * S)],
                   fill=(255, 255, 255, 255))
    y = by + 1.5 + 6
    if subtitle:
        y = _text(canvas, subtitle, bx, y, 10, 12, S, kind="body",
                  fill=(255, 255, 255, 255), max_w=block_w, max_lines=2) + 6
    _text(canvas, BYLINE, bx, y, 7, 8.4, S, kind="sans",
          fill=(255, 255, 255, 255), max_w=block_w, max_lines=1)


def _sec1_brush(canvas, H, S) -> None:
    # width fixed (canvas width is constant), height scales with H — matches
    # the Figma container-query sizing at every ratio
    _deco(canvas, "sec1_brush.png", LIME, 158, 0.0148 * H, 112.75, 0.2125 * H, S)


def _sec1_zigzag(canvas, H, S, ratio_id, rgb=LIME) -> None:
    """Scribble 4 (bottom-left zigzag). Figma bakes a different stretch, skew
    and rotation per ratio, so each asset is a full-canvas 4x recomposition
    (see scripts' bake notes) pasted 1:1 over the canvas."""
    _deco(canvas, f"zigzag_{_TAG[ratio_id]}.png", rgb, 0, 0, W, H, S)


def _r_sec1a(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    _sec1_base(canvas, photo, H, S)
    _sec1_centre(canvas, title, subtitle, H, S, 160)
    _sec1_brush(canvas, H, S)


def _r_sec1b(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    _sec1_base(canvas, photo, H, S)
    _sec1_centre(canvas, title, subtitle, H, S, 153)
    _sec1_zigzag(canvas, H, S, ratio_id)


def _r_sec1c(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    _sec1_base(canvas, photo, H, S)
    _sec1_centre(canvas, title, subtitle, H, S, 153)
    _deco(canvas, "sec1_claw.png", LIME, 211, 0.271 * H, 54, 0.174 * H, S)


def _r_sec1d(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    """Style 4 — logo top-left (30,26), text block bottom-left (inset 30/28)."""
    draw = _sec1_base(canvas, photo, H, S)
    _deco(canvas, "cj_logo.png", WHITE, 30, 26, 62.4, 26, S)
    n = _n_lines(canvas, title, 26, S, 210, 2)
    m = _n_lines(canvas, subtitle, 8, S, 202, 2, "body") if subtitle else 0
    block_h = 31.2 * n + 5 + 1.5 + 6 + m * 9.6
    top = H - 28 - block_h
    _text(canvas, title, 30, top, 26, 31.2, S,
          fill=(255, 255, 255, 255), max_w=210, max_lines=2)
    by = top + 31.2 * n + 5
    draw.rectangle([round(30 * S), round(by * S),
                    round(240 * S), round((by + 1.2) * S)],
                   fill=(255, 255, 255, 255))
    if subtitle:
        _text(canvas, subtitle, 30, by + 1.5 + 6, 8, 9.6, S, kind="body",
              fill=(255, 255, 255, 255), max_w=202, max_lines=2)
    _sec1_brush(canvas, H, S)


# ── Secondary 2 / 3 / 4 — photo, 40% shade + centred title + decoration ───────
SEC_DECO = {"green": GREEN, "lime": LIME, "white": WHITE}
SEC_TITLE = {"green": WHITE, "lime": LIME, "white": WHITE}


def _sec_title(canvas, photo, title, H, S, theme, lh) -> None:
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    if not title:
        return
    canvas.alpha_composite(Image.new("RGBA", canvas.size, (0, 0, 0, 102)))
    n = _n_lines(canvas, title, 26, S, 198, 2)
    _text(canvas, title, 36, (H - lh * n) / 2, 26, lh, S,
          fill=SEC_TITLE[theme] + (255,), max_w=198, max_lines=2, align="center")


def _r_sec2(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    """Band frame: canvas edge down to a rounded cutout at inset 10 (radius 10
    on the TL/BR corners only) — geometry unchanged from the previous set."""
    _sec_title(canvas, photo, title, H, S, theme, 31.2)
    mask = Image.new("L", canvas.size, 255)
    ImageDraw.Draw(mask).rounded_rectangle(
        [round(10 * S), round(10 * S),
         round((W - 10) * S) - 1, round((H - 10) * S) - 1],
        radius=round(10 * S), fill=0, corners=(True, False, True, False))
    band = Image.new("RGBA", canvas.size, SEC_DECO[theme] + (255,))
    band.putalpha(mask)
    canvas.alpha_composite(band)


def _r_sec3(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    """Hand-drawn double-line frame (per-ratio 4x export, centred)."""
    _sec_title(canvas, photo, title, H, S, theme, 31.2)
    fr = _recolor(_asset(f"sec3_frame_{_TAG[ratio_id]}.png"), SEC_DECO[theme])
    fw, fh = fr.width / 4, fr.height / 4
    fr = fr.resize((round(fw * S), round(fh * S)), Image.LANCZOS)
    canvas.alpha_composite(fr, (round((W - fw) / 2 * S), round((H - fh) / 2 * S)))


_SEC4_ARROW_TOP = {"1:1": 0.7407, "4:5": 0.7407, "9:16": 0.7667}
_SEC4_BRUSH_Y = {"1:1": -13, "4:5": 0, "9:16": -6}


def _r_sec4a(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    _sec_title(canvas, photo, title, H, S, theme, 26)
    _deco(canvas, "sec4_brush.png", SEC_DECO[theme],
          151, _SEC4_BRUSH_Y[ratio_id], 119.5, 74, S)
    # same Scribble 4 as Secondary 1 style 2 (identical container in Figma)
    _sec1_zigzag(canvas, H, S, ratio_id, rgb=SEC_DECO[theme])


def _r_sec4b(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    _sec_title(canvas, photo, title, H, S, theme, 26)
    _deco(canvas, "sec4_claw.png", SEC_DECO[theme], 190, 6, 73.25, 81, S)
    _deco(canvas, "sec4_arrow.png", SEC_DECO[theme],
          0, _SEC4_ARROW_TOP[ratio_id] * H, 100.5, 110, S)


# ── Image Only — confetti overlay (272×478 art, offset per ratio) ─────────────
_CONFETTI_BOX = {
    "1:1": (-1, -104, 272, 478),
    "4:5": (-1, -43, 272, 478),
    "9:16": (-4, -5, 278, 489),
}
_CONFETTI_TONE = {"green": "dark", "lime": "light", "white": "light"}


def _r_imageonly(canvas, photo, title, subtitle, H, S, theme, ratio_id):
    canvas.paste(_cover(photo, round(W * S), round(H * S)), (0, 0))
    x, y, w, h = _CONFETTI_BOX[ratio_id]
    ov = _asset(f"io_confetti_{_CONFETTI_TONE[theme]}.png") \
        .resize((round(w * S), round(h * S)), Image.LANCZOS)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(ov, (round(x * S), round(y * S)))
    canvas.alpha_composite(layer)


# ── Template registry ──────────────────────────────────────────────────────────
# category: cover | text | secondary | image
# requiresImage / supportsTitle / supportsSubtitle drive the frontend's
# sort-compatible-first + dim-incompatible behaviour.
def _tpl(id, name, category, render, *, image=True, title=True, sub=True):
    return {"id": id, "name": name, "category": category, "render": render,
            "requiresImage": image, "supportsTitle": title,
            "supportsSubtitle": sub}

TEMPLATES = [
    _tpl("cover1", "Cover 1", "cover", _r_cover1),
    _tpl("cover2", "Cover 2", "cover", _r_cover2),
    _tpl("cover3", "Cover 3", "cover", _r_cover3),
    _tpl("cover4", "Cover 4", "cover", _r_cover4),
    _tpl("cover5", "Cover 5", "cover", _r_cover5),
    _tpl("cover6", "Cover 6", "cover", _r_cover6),
    _tpl("cover7", "Cover 7", "cover", _r_cover7),
    _tpl("textonly", "Text Only", "text", _r_textonly, image=False),
    _tpl("sec1a", "Secondary 1·1", "secondary", _r_sec1a),
    _tpl("sec1b", "Secondary 1·2", "secondary", _r_sec1b),
    _tpl("sec1c", "Secondary 1·3", "secondary", _r_sec1c),
    _tpl("sec1d", "Secondary 1·4", "secondary", _r_sec1d),
    _tpl("sec2", "Secondary 2", "secondary", _r_sec2, sub=False),
    _tpl("sec3", "Secondary 3", "secondary", _r_sec3, sub=False),
    _tpl("sec4a", "Secondary 4·1", "secondary", _r_sec4a, sub=False),
    _tpl("sec4b", "Secondary 4·2", "secondary", _r_sec4b, sub=False),
    _tpl("imageonly", "Image Only", "image", _r_imageonly, title=False, sub=False),
]
TEMPLATE_BY_ID = {t["id"]: t for t in TEMPLATES}

TESTING_STYLES = {t["id"]: t["render"] for t in TEMPLATES}  # main.py validation
TESTING_THEMES = ["green", "lime", "white"]


def template_meta() -> list:
    """JSON-safe metadata for the frontend (render fns stripped)."""
    return [{k: t[k] for k in
             ("id", "name", "category", "requiresImage",
              "supportsTitle", "supportsSubtitle")} for t in TEMPLATES]


# ── render + cache ─────────────────────────────────────────────────────────────
_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_CACHE_MAX = 96


def render_testing(style: str, theme: str, ratio_id: str, img_bytes: bytes,
                   title: str, subtitle: str, out_w: int) -> tuple:
    """Render one template+theme+ratio; returns (png_bytes, title_truncated).
    Memoised on all inputs. Emoji are stripped (the template fonts have no
    glyphs for them). An empty title falls back to the brand name (only on
    templates with a title slot); an empty subtitle is simply omitted."""
    global _TRUNCATED
    tpl = TEMPLATE_BY_ID[style]
    title = _strip_emoji((title or "").strip())
    subtitle = _strip_emoji((subtitle or "").strip())
    if not tpl["supportsTitle"]:
        title, subtitle = "", ""
    elif not title:
        title = "Construction Junction"
    if not tpl["supportsSubtitle"]:
        subtitle = ""

    key = ("social2", style, theme, ratio_id,
           hashlib.sha1(img_bytes).hexdigest(), title, subtitle, out_w)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    H = H_BY_RATIO[ratio_id]
    S = out_w / W
    photo = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    canvas = Image.new("RGBA", (out_w, round(H * S)), (255, 255, 255, 255))
    _TRUNCATED = False
    tpl["render"](canvas, photo, title, subtitle, H, S, theme, ratio_id)
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    result = (buf.getvalue(), _TRUNCATED)

    _CACHE[key] = result
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return result
