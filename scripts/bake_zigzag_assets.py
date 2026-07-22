"""Fit ONE affine (art -> canvas, 6 params) per ratio for Scribble 4 against
the 1x ground-truth set screenshots, then bake 4x overlays."""
import math
import numpy as np
from PIL import Image, ImageFilter
from scipy.optimize import minimize

DIR = "/private/tmp/claude-501/-Users-cynthialu-cj-listing-formatter/9b804b32-902a-4f77-97f5-2220242c4559/scratchpad/figma"
DST = "/Users/cynthialu/cj-listing-formatter/static/social2"
LIME = (188, 240, 14)
W = 270

CASES = {
    "11":  {"H": 270.0, "Hfig": 270, "rot": 15.36, "tile": (310, 20)},
    "45":  {"H": 337.5, "Hfig": 339, "rot": 19.03, "tile": (310, 310)},
    "916": {"H": 480.0, "Hfig": 480, "rot": 26.03, "tile": (310, 669)},
}

art = Image.open(f"{DIR}/assets/zz_mask_916.png").getchannel("A")
AW, AH = art.size
fig = Image.open(f"{DIR}/secondary1.png").convert("RGB")


def warp(params, H, scale=1.0):
    """params = (a,b,tx,c,d,ty): canvas(units) = A @ art_px + t. Render at
    `scale` px/unit; PIL needs inverse (canvas->art)."""
    a, b, tx, c, d, ty = params
    A = np.array([[a, b], [c, d]])
    Ainv = np.linalg.inv(A)
    # canvas px -> units -> art px: art = Ainv @ (canvas_px/scale - t)
    ia, ib = Ainv[0] / scale, Ainv[1] / scale
    data = (ia[0], ia[1], -(Ainv[0] @ [tx, ty]),
            ib[0], ib[1], -(Ainv[1] @ [tx, ty]))
    return art.transform((round(W * scale), round(H * scale)), Image.AFFINE,
                         data, resample=Image.BILINEAR)


def tile_alpha(p):
    tx, ty = p["tile"]
    crop = np.asarray(fig.crop((tx, ty, tx + 270, ty + p["Hfig"]))).astype(int)
    d = np.sqrt(((crop - np.array(LIME)) ** 2).sum(axis=2))
    return np.clip(1 - d / 110, 0, 1)


def soft_iou(a, b):
    return np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-6)


for tag, p in CASES.items():
    target = tile_alpha(p)
    Ht = p["Hfig"]
    ys, xs = np.where(target > 0.5)
    t_cx, t_cy = xs.mean(), ys.mean()
    a_ink = art.getbbox()
    a_cx, a_cy = (a_ink[0] + a_ink[2]) / 2, (a_ink[1] + a_ink[3]) / 2

    best = None
    th = math.radians(p["rot"])
    for sx in (0.22, 0.28, 0.34, 0.42):
        for sy in (0.3, 0.45, 0.6, 0.8):
            for rs in (th, -th):
                R = np.array([[math.cos(rs), -math.sin(rs)],
                              [math.sin(rs), math.cos(rs)]])
                A = R @ np.diag([sx, sy])
                t = np.array([t_cx, t_cy]) - A @ [a_cx, a_cy]
                x0 = np.array([A[0, 0], A[0, 1], t[0], A[1, 0], A[1, 1], t[1]])

                def loss(v):
                    try:
                        w = np.asarray(warp(v, Ht)).astype(float) / 255
                    except Exception:
                        return 1.0
                    return 1 - soft_iou(w, target)

                r = minimize(loss, x0, method="Nelder-Mead",
                             options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
                if best is None or r.fun < best[0]:
                    best = (r.fun, r.x)
    print(tag, "IoU %.3f" % (1 - best[0]), np.round(best[1], 3))
    p["affine"] = best[1]

# bake at 4x, engine H
for tag, p in CASES.items():
    canvas = warp(p["affine"], p["H"], scale=4.0)
    out = Image.merge("RGBA", [Image.new("L", canvas.size, 255)] * 3 + [canvas])
    out.save(f"{DST}/zigzag_{tag}.png")
    a = np.asarray(canvas)
    ys, xs = np.where(a > 100)
    print(tag, "baked", out.size, "ink",
          [round(v / 4, 1) for v in (xs.min(), ys.min(), xs.max(), ys.max())])
