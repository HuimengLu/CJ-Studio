"""One-time preparation of Social template assets exported from Figma.

The Figma MCP asset endpoint flattens PNG exports onto a solid background
(white #ffffff or canvas grey #f5f5f5), losing the alpha channel. Every
decoration is a single solid colour, so the alpha can be recovered exactly by
inverting the flatten equation  rendered = a*C + (1-a)*B :

    a = ((B - P) . (B - C)) / |B - C|^2

where P is the rendered pixel, C the decoration colour and B the background.

Inputs (raw exports, committed): static/social/s*_*.png, cj_logo_white.png
Outputs: static/social/proc_*.png (RGBA, used by social.py at runtime)

Run from the repo root:  python3 scripts/prep_social_assets.py
"""
import os

import numpy as np
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "static", "social")

LIME = (188, 240, 14)      # CJ accent scribbles
GREEN = (0, 86, 24)        # CJ brand green #005618
WHITE = (255, 255, 255)
BG_WHITE = (255, 255, 255)
BG_GREY = (245, 245, 245)  # Figma canvas grey used by some exports


def unmix(src: str, color: tuple, bg: tuple, dst: str) -> None:
    """Recover alpha of a solid-`color` decoration flattened onto `bg`."""
    p = np.asarray(Image.open(os.path.join(BASE, src)).convert("RGB"), np.float32)
    c = np.array(color, np.float32)
    b = np.array(bg, np.float32)
    denom = float(((c - b) ** 2).sum())
    a = ((b - p) * (b - c)).sum(axis=2) / denom
    a = np.clip(a, 0.0, 1.0)
    out = np.zeros(p.shape[:2] + (4,), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color
    out[..., 3] = (a * 255).round().astype(np.uint8)
    Image.fromarray(out).save(os.path.join(BASE, dst))
    print(f"{src} -> {dst}  coverage={a.mean():.3f}")


def main() -> None:
    unmix("s1_scribble.png", LIME, BG_WHITE, "proc_s1_scribble.png")
    unmix("s2_scribble.png", LIME, BG_WHITE, "proc_s2_scribble.png")
    unmix("s3_scribble.png", LIME, BG_WHITE, "proc_s3_scribble.png")
    unmix("s4_scribble.png", LIME, BG_GREY, "proc_s4_scribble.png")
    unmix("s5_scribble.png", LIME, BG_GREY, "proc_s5_scribble.png")
    unmix("s3_mask.png", GREEN, BG_WHITE, "proc_s3_blob.png")   # blob backdrop + photo mask
    unmix("s4_panel.png", GREEN, BG_GREY, "proc_s4_panel.png")
    unmix("s5_frame.png", WHITE, BG_GREY, "proc_s5_frame.png")
    unmix("cj_logo_white.png", WHITE, BG_GREY, "proc_logo_white.png")


if __name__ == "__main__":
    main()
