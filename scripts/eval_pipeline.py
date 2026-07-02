"""
Batch before/after pipeline evaluation.

Usage:
    python scripts/eval_pipeline.py --input test_images/ --output eval_out/ [--model u2net]

Processes every JPG/PNG/WEBP in --input, saves side-by-side before/after PNGs
to --output for visual inspection.  Pass --model to compare different rembg
models (e.g. u2net vs birefnet-general).

Available rembg models worth comparing:
  u2net              173 MB  current default — general purpose
  isnet-general-use  ~170 MB newer architecture, better product shots
  birefnet-general   ~400 MB highest quality; noticeably cleaner masks
  birefnet-general-lite ~170 MB faster BiRefNet variant
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ── allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_session(model: str):
    from rembg import new_session
    print(f"Loading model '{model}' …", flush=True)
    t0 = time.perf_counter()
    s = new_session(model)
    print(f"  ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return s


def remove_bg(pil_img: Image.Image, session) -> tuple[np.ndarray, np.ndarray]:
    """Run rembg and return (rgb, alpha) uint8 arrays."""
    from rembg import remove as rembg_remove
    rgba = rembg_remove(pil_img.convert("RGBA"), session=session)
    arr = np.array(rgba)
    return arr[:, :, :3], arr[:, :, 3]


def tighten_alpha(alpha: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(alpha, 15, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)
    inv = cv2.bitwise_not(binary)
    _, labels = cv2.connectedComponents(inv, connectivity=8)
    border_labels = set(np.concatenate([
        labels[0, :], labels[-1, :], labels[1:-1, 0], labels[1:-1, -1],
    ]).tolist())
    filled = binary.copy()
    for lbl in np.unique(labels):
        if lbl != 0 and lbl not in border_labels:
            filled[labels == lbl] = 255
    out = np.where(filled > 0, np.where(alpha > 0, alpha, np.uint8(255)), np.uint8(0)).astype(np.uint8)
    out[out < 80] = 0
    return out


def color_guided_cleanup(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    b = max(10, min(30, h // 20, w // 20))
    strips = [rgb[:b, :], rgb[-b:, :], rgb[:, :b], rgb[:, -b:]]
    border_px = np.concatenate([s.reshape(-1, 3) for s in strips]).astype(np.float32)
    if not len(border_px):
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


def erode_mask(alpha: np.ndarray, radius: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(alpha, k, iterations=1)


def decontaminate_edges(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    b = max(10, min(30, h // 20, w // 20))
    strips = [rgb[:b, :], rgb[-b:, :], rgb[:, :b], rgb[:, -b:]]
    border_px = np.concatenate([s.reshape(-1, 3) for s in strips]).astype(np.float32)
    if not len(border_px):
        return rgb
    bg = np.median(border_px, axis=0)
    if float(np.mean(bg)) > 245:
        return rgb
    edge = (alpha > 0) & (alpha < 254)
    if not edge.any():
        return rgb
    a_col = alpha[edge].astype(np.float32)[:, np.newaxis] / 255.0
    c_vals = rgb[edge].astype(np.float32)
    fg = (c_vals - bg[np.newaxis, :] * (1.0 - a_col)) / np.clip(a_col, 0.01, 1.0)
    out = rgb.copy()
    out[edge] = np.clip(fg, 0, 255).astype(np.uint8)
    return out


CANVAS_BG = (224, 221, 211)


def composite_on_canvas(rgb: np.ndarray, alpha: np.ndarray, size: int = 512) -> Image.Image:
    """Place extracted subject centred on a neutral canvas for comparison."""
    h, w = rgb.shape[:2]
    scale = (size * 0.80) / max(h, w, 1)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    r_rgb  = cv2.resize(rgb,   (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    r_mask = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((size, size, 3), CANVAS_BG, dtype=np.uint8)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    a_f = r_mask.astype(np.float32) / 255.0
    roi = canvas[oy:oy+nh, ox:ox+nw].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = r_rgb[:, :, c] * a_f + roi[:, :, c] * (1.0 - a_f)
    canvas[oy:oy+nh, ox:ox+nw] = roi.astype(np.uint8)
    return Image.fromarray(canvas)


def label(img: Image.Image, text: str, bg: tuple = (40, 40, 40)) -> Image.Image:
    """Burn a small text label into the top-left corner."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, len(text) * 7 + 8, 18], fill=bg)
    d.text((4, 2), text, fill=(255, 255, 255))
    return out


def make_comparison(original: Image.Image, rgb_raw: np.ndarray, alpha_raw: np.ndarray,
                    rgb_clean: np.ndarray, alpha_clean: np.ndarray,
                    model: str, elapsed: float) -> Image.Image:
    """Return a 4-panel horizontal strip: original | raw mask | raw composite | clean composite."""
    SZ = 512
    orig_sq = ImageOps.pad(original, (SZ, SZ), color=CANVAS_BG)

    raw_alpha_img  = Image.fromarray(alpha_raw).convert("RGB")
    clean_alpha_img = Image.fromarray(alpha_clean).convert("RGB")
    raw_alpha_sq   = ImageOps.pad(raw_alpha_img,   (SZ, SZ), color=(128,)*3)
    clean_alpha_sq = ImageOps.pad(clean_alpha_img, (SZ, SZ), color=(128,)*3)

    raw_composite   = composite_on_canvas(rgb_raw,   alpha_raw,   SZ)
    clean_composite = composite_on_canvas(rgb_clean, alpha_clean, SZ)

    panels = [
        label(orig_sq,         "original"),
        label(raw_alpha_sq,    f"mask [{model}]"),
        label(raw_composite,   f"before post-proc  {elapsed:.1f}s"),
        label(clean_composite, "after decontam+erode"),
    ]
    strip = Image.new("RGB", (SZ * len(panels), SZ), (20, 20, 20))
    for i, p in enumerate(panels):
        strip.paste(p, (i * SZ, 0))
    return strip


def process_image(path: Path, session, model: str) -> Image.Image:
    original = ImageOps.exif_transpose(Image.open(path))
    t0 = time.perf_counter()
    rgb_raw, alpha_raw = remove_bg(original, session)
    elapsed = time.perf_counter() - t0

    # Raw compositing for the "before" panel
    alpha_post = tighten_alpha(alpha_raw.copy())
    alpha_post = color_guided_cleanup(rgb_raw, alpha_post)

    # Clean pipeline ("after" panel)
    alpha_clean = erode_mask(alpha_post.copy(), radius=1)
    rgb_clean   = decontaminate_edges(rgb_raw.copy(), alpha_clean)

    return make_comparison(original, rgb_raw, alpha_post, rgb_clean, alpha_clean, model, elapsed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  default="test_images", help="Directory of input images")
    ap.add_argument("--output", default="eval_out",    help="Directory for comparison outputs")
    ap.add_argument("--model",  default="u2net",       help="rembg model name")
    ap.add_argument("--model2", default=None,          help="Second model to compare (optional)")
    args = ap.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)

    if not images:
        print(f"No images found in {in_dir}. Add JPG/PNG/WEBP files there and re-run.")
        sys.exit(1)

    print(f"Found {len(images)} image(s). Output → {out_dir}/")

    session1 = build_session(args.model)
    session2 = build_session(args.model2) if args.model2 else None

    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}", flush=True)
        try:
            strip1 = process_image(path, session1, args.model)
            if session2:
                strip2 = process_image(path, session2, args.model2)
                combined = Image.new("RGB", (strip1.width, strip1.height + strip2.height + 4), (20, 20, 20))
                combined.paste(strip1, (0, 0))
                combined.paste(strip2, (0, strip1.height + 4))
                combined.save(out_dir / f"{path.stem}_compare.png")
            else:
                strip1.save(out_dir / f"{path.stem}_eval.png")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nDone. {len(images)} comparison(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
