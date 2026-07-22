"""Process raw Figma MCP exports into static/social2/ PNGs (new template set).

Point SRC at a directory of 4x `download_assets` exports from the Funnnn
Figma file (node ids are recorded in backend/social_engine.py's docstring and
git history). Figma's MCP export flattens every layer onto a flat backdrop
(light-gray #F5F5F5, white, or the parent theme colour), so decorations are
chroma-keyed back out: solve  pixel = a*deco + (1-a)*backdrop  per channel for
the known deco colour, producing a white RGBA mask the engine recolours at
render time. The Cover-7 plaque masks come from the context SVGs, rasterised
separately (svgpathtools + PIL); the confetti overlays ship as-is from the
raw source images.
"""
import os
import numpy as np
from PIL import Image

SRC = os.environ.get("SOCIAL2_SRC", os.path.dirname(os.path.abspath(__file__)) + "/assets")
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "static", "social2")
os.makedirs(DST, exist_ok=True)

LIME = (188, 240, 14)
GREEN = (0, 86, 24)
WHITE = (255, 255, 255)
GRAY = (245, 245, 245)


def unmix(src, deco, backdrop=None, out=None, trim=False):
    """Key `deco` colour off a flat `backdrop`; save white+alpha RGBA.
    backdrop=None auto-detects it as the image's dominant colour."""
    im = Image.open(f"{SRC}/{src}").convert("RGB")
    if backdrop is None:
        from collections import Counter
        backdrop = Counter(im.resize((80, 80)).getdata()).most_common(1)[0][0]
    a = np.asarray(im).astype(float)
    ws, alphas = [], []
    thresh = 60 if any(abs(deco[c] - backdrop[c]) >= 60 for c in range(3)) else 8
    for c in range(3):
        d = deco[c] - backdrop[c]
        if abs(d) < thresh:
            continue
        alphas.append((a[:, :, c] - backdrop[c]) / d)
        ws.append(abs(d))
    alpha = sum(w * al for w, al in zip(ws, alphas)) / sum(ws)
    alpha = np.clip(alpha * 255, 0, 255).astype("uint8")
    rgba = np.dstack([np.full_like(alpha, 255)] * 3 + [alpha])
    img = Image.fromarray(rgba, "RGBA")
    box = None
    if trim:
        box = Image.fromarray(alpha, "L").point(lambda v: v if v > 24 else 0).getbbox()
        img = img.crop(box)
    img.save(f"{DST}/{out}")
    print(out, img.size, "trim" if trim else "", box or "")
    return box, Image.open(f"{SRC}/{src}").size


# ── recolourable single-colour decorations (white+alpha masks) ─────────────────
unmix("c1_squiggle.png", LIME, None, out="squiggle.png")            # title underline (Cover 1/4/7)
unmix("c2_arrow.png", LIME, GRAY, out="c2_arrow.png", trim=True)     # Cover 2 arrow
unmix("c5_claw.png", LIME, GRAY, out="c5_claw.png", trim=True)       # Cover 5 claw scribble
unmix("cj_logo_2025.png", WHITE, GREEN, out="cj_logo.png")           # CJ 2025 logo

for tag in ("11", "45", "916"):
    for sub in ("sub", "nosub"):
        # short titles: white blob on gray · long titles: green blob on gray
        unmix(f"c2_sub_{tag}_{sub}_s.png", WHITE, GRAY, out=f"c2_blob_{tag}_{sub}_1.png")
        unmix(f"c2_sub_{tag}_{sub}_l.png", GREEN, GRAY, out=f"c2_blob_{tag}_{sub}_2.png")
    unmix(f"sec3_frame_{tag}_lime.png", LIME, GRAY, out=f"sec3_frame_{tag}.png")
    # plaque masks were rasterised from the context SVGs (already clean alpha)
    m = Image.open(f"{SRC}/c7_plaque_{tag}_mask.png").convert("L")
    rgba = np.dstack([np.full((m.size[1], m.size[0]), 255, dtype="uint8")] * 3 + [np.asarray(m)])
    Image.fromarray(rgba, "RGBA").save(f"{DST}/c7_plaque_{tag}.png")
    print(f"c7_plaque_{tag}.png", m.size)

unmix("sec1_scr1.png", LIME, None, out="sec1_claw.png", trim=True)     # Sec-1 style 3
unmix("sec1_scr3.png", LIME, GRAY, out="sec1_brush.png", trim=True)    # Sec-1 styles 1+4 (top-right)
# Scribble 4 (bottom-left zigzag, Sec-1 style 2 + Sec-4 style 1): Figma's node
# export drops the mask (solid wedge) and its CSS mask params don't reproduce
# the render, so zigzag_{11,45,916}.png are baked separately: one affine
# (squiggle art px -> canvas units) per ratio, fitted with Nelder-Mead against
# the 1x set screenshots (soft-IoU 0.93-0.95; ink bbox matches Figma +-2u).
#   art: Rectangle-7 mask export, 992x208 (zz_mask_916.png)
#   affine (a b tx / c d ty), canvas = A @ art_px + t, engine H per ratio:
#   1:1  ( 0.266 -0.090  -23.015 / 0.092 0.263 179.589)
#   4:5  ( 0.334 -0.113  -54.254 / 0.115 0.330 225.453)
#   9:16 ( 0.473 -0.160 -118.514 / 0.162 0.465 319.856)
# The engine pastes them full-canvas. (Bake code: git history, fit_affine.py.)
unmix("sec4_s1_brush.png", LIME, GRAY, out="sec4_brush.png", trim=True)
unmix("sec4_s2_claw.png", LIME, GRAY, out="sec4_claw.png", trim=True)
unmix("sec4_s2_arrow.png", LIME, GRAY, out="sec4_arrow.png", trim=True)

# ── confetti overlays (multi-colour, shipped as-is from the raw source) ────────
for tone in ("light", "dark"):
    im = Image.open(f"{SRC}/io_raw_{tone}_1.png")
    im.save(f"{DST}/io_confetti_{tone}.png")
    print(f"io_confetti_{tone}.png", im.size)

# sanity proof: composite a few masks on green
proof = Image.new("RGB", (1200, 400), (0, 86, 24))
x = 10
for name in ("squiggle", "c2_arrow", "c5_claw", "cj_logo", "sec1_brush",
             "sec1_zigzag", "sec1_claw", "sec4_arrow", "sec3_frame_45"):
    im = Image.open(f"{DST}/{name}.png")
    im.thumbnail((160, 320))
    proof.paste(im, (x, 20), im)
    x += im.width + 10
proof.save(f"{SRC}/proof_masks.png")
print("proof written")
