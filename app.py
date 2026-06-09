import functools
import http.server
import io
import logging
import os
import socket
import threading
import time

import cv2
import numpy as np
import streamlit as st

from PIL import Image, ImageEnhance, ImageOps

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cj_listing")

# ── rembg model selection ──────────────────────────────────────────────────────
# isnet-general-use  ~170 MB  default – DIS architecture, better edges on products
# birefnet-general   ~400 MB  best quality; override via env var below
# u2net              ~173 MB  original baseline (kept for reference)
_REMBG_MODEL: str = os.environ.get("CJ_REMBG_MODEL", "isnet-general-use")

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CJ Listing Formatter",
    layout="centered",
    initial_sidebar_state="collapsed",
)

PRIMARY = "#C4D938"
DARK = "#555555"

st.markdown(
    f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: #f8f8f5; }}
  .block-container {{ max-width: 880px; padding-top: 3rem; padding-bottom: 3rem; }}
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
  .divider {{
    border: none;
    border-top: 1px solid #e8e8e4;
    margin: 1.5rem 0;
  }}
  /* ── Hide sidebar & collapse arrow ── */
  [data-testid="stSidebar"],
  [data-testid="collapsedControl"] {{
    display: none !important;
  }}
  [data-testid="stStatusWidget"] {{ display: none !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# ── rembg session (loaded once, cached for the lifetime of the server) ─────────

@st.cache_resource(show_spinner=False)
def _rembg_session():
    from rembg import new_session

    log.info("Loading rembg model '%s' …", _REMBG_MODEL)
    t0 = time.perf_counter()
    try:
        session = new_session(_REMBG_MODEL)
    except Exception as exc:
        log.error("Failed to load rembg model '%s': %s", _REMBG_MODEL, exc, exc_info=True)
        raise RuntimeError(
            f'Could not load background-removal model "{_REMBG_MODEL}". '
            "Check your internet connection — the first run downloads the model weights. "
            f"Detail: {exc}"
        ) from exc

    log.info("rembg session ready in %.1f s", time.perf_counter() - t0)
    return session


# ── image processing ───────────────────────────────────────────────────────────

def _extract_subject(pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove background with rembg.  Returns (rgb, alpha) as uint8 numpy arrays.

    Uses post_process_mask=True to run rembg's built-in mask cleanup (morphological
    open/close + connected-component filtering).  No additional hand-rolled
    post-processing is needed or applied on top of this.
    """
    from rembg import remove as rembg_remove

    log.info(
        "Removing background: %dx%d  model=%s",
        pil_img.width, pil_img.height, _REMBG_MODEL,
    )
    t0 = time.perf_counter()

    try:
        session = _rembg_session()
        rgba = rembg_remove(
            pil_img.convert("RGBA"),
            session=session,
            post_process_mask=True,
        )
    except RuntimeError:
        raise
    except MemoryError as exc:
        log.error("OOM during rembg inference: %s", exc, exc_info=True)
        raise RuntimeError(
            "Not enough memory to process this image. "
            "Try a smaller image (< 4000 × 4000 px)."
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

    shadow = cv2.GaussianBlur(shadow, (0, 0), 20.0)
    shadow = np.clip(shadow * 0.22, 0.0, 1.0)
    return shadow, empty


def make_listing(pil_img: Image.Image):
    """
    Full pipeline: rembg subject extraction → crop → 1600×1600 canvas + shadow.
    Returns (result_pil, warning_str | None).
    On rembg failure returns (None, user-friendly error string) instead of raising.
    """
    log.info(
        "make_listing: start  input=%dx%d",
        pil_img.width, pil_img.height,
    )
    t_total = time.perf_counter()

    try:
        rgb, alpha = _extract_subject(pil_img)
    except RuntimeError as exc:
        log.warning("make_listing: subject extraction failed — %s", exc)
        return None, str(exc)

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

    # ── Canvas ────────────────────────────────────────────────────────────────
    CS = 1600
    canvas = np.full((CS, CS, 3), [224, 221, 211], dtype=np.uint8)  # #E0DDD3

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

    # ── Composite item ────────────────────────────────────────────────────────
    alpha_f = r_mask[img_y, img_x].astype(np.float32) / 255.0
    roi = canvas[can_y, can_x].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = r_rgb[img_y, img_x, c] * alpha_f + roi[:, :, c] * (1.0 - alpha_f)
    canvas[can_y, can_x] = roi.astype(np.uint8)

    # Subtle brightness + contrast lift
    out = Image.fromarray(canvas)
    out = ImageEnhance.Brightness(out).enhance(1.02)
    out = ImageEnhance.Contrast(out).enhance(1.05)

    log.info(
        "make_listing: done in %.2f s  output=%dx%d",
        time.perf_counter() - t_total, out.width, out.height,
    )
    return out, None


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_RATIO_AR = {"1:1": 1.0, "4:3": 4/3, "4:5": 4/5}


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
for _k in ("original", "result", "warn", "_cj_pending"):
    if _k not in st.session_state:
        st.session_state[_k] = None
if "ratio" not in st.session_state:
    st.session_state.ratio = "1:1"
if "_cj_loading" not in st.session_state:
    st.session_state._cj_loading = False

# ── Shared nav component ──────────────────────────────────────────────────────
# Single source-of-truth: same CSS is embedded in both the Streamlit page and
# the iaac static page.  Only positioning differs (iaac is fixed-overlay).
_NAV_SHARED_CSS = """\
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

# ── Embedded iaac HTTP server (same process, correct MIME types) ──────────────
_IAAC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "iaac")

@st.cache_resource(show_spinner=False)
def _iaac_port() -> int:
    """Start an embedded HTTP server for the iaac static app.
    Injects a Listing/Social nav pill into index.html.
    Returns the port it's listening on."""

    # iaac needs the shared CSS + a fixed-position wrapper so the pill floats
    # over the existing iaac page layout.
    _NAV_CSS = (
        "<style>\n"
        + _NAV_SHARED_CSS
        + "\n#cj-nav-wrap{position:fixed;top:14px;left:16px;z-index:99999}"
        + "\n</style>"
    )

    _NAV_HTML = (
        '<div id="cj-nav-wrap"><div id="cj-nav">\n'
        '  <a class="cj-np" href="http://localhost:8501/">Listing</a>\n'
        '  <span class="cj-np active">Social</span>\n'
        "</div></div>"
    )

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=_IAAC_DIR, **kw)

        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(_IAAC_DIR, "index.html"), encoding="utf-8") as f:
                    html = f.read()
                html = html.replace("</head>", _NAV_CSS + "\n</head>")
                html = html.replace("</body>", _NAV_HTML + "\n</body>")
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

    with socket.socket() as _s:
        _s.bind(("", 0))
        _port = _s.getsockname()[1]

    _srv = http.server.HTTPServer(("localhost", _port), _Handler)
    threading.Thread(target=_srv.serve_forever, daemon=True).start()
    return _port

# Start the iaac server once; cached for the lifetime of the Streamlit process
_IAAC_PORT = _iaac_port()

# Nav pill — reuses the same _NAV_SHARED_CSS as the iaac page.
# On the Streamlit page the pill sits in normal document flow (no fixed position).
st.markdown(
    f"<style>{_NAV_SHARED_CSS}</style>"
    f'<div id="cj-nav">'
    f'  <span class="cj-np active">Listing</span>'
    f'  <a class="cj-np" href="http://localhost:{_IAAC_PORT}/">Social</a>'
    f"</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE · Item Listing (CJ Formatter)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
        f"<h2 style='color:{DARK};margin:0 0 1.4rem;font-size:1.85rem;font-weight:700;'>"
        "CJ Listing Formatter</h2>",
        unsafe_allow_html=True,
    )

# ── upload screen ──────────────────────────────────────────────────────────
if st.session_state.result is None:
    _is_loading = st.session_state._cj_loading

    # File uploader — always rendered
    _uploaded = st.file_uploader(
        "Drop photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<p style='text-align:center;color:#bbb;font-size:0.78rem;"
        "margin-top:-0.3rem;'>JPG · PNG · WEBP · up to 200 MB</p>",
        unsafe_allow_html=True,
    )

    if not _is_loading and st.session_state.warn:
        st.markdown(
            f"<div class='warn-box'>&#9888; {st.session_state.warn}</div>",
            unsafe_allow_html=True,
        )

    # ── Loading modal — shown as a full-screen popup while processing ──────
    if _is_loading:
        st.markdown(
            """
            <style>
            .loader {
              width: 35px;
              aspect-ratio: 1;
              --c:no-repeat linear-gradient(#046D8B 0 0);
              background:
                var(--c) 0 0,
                var(--c) 100% 0,
                var(--c) 100% 100%,
                var(--c) 0 100%;
              animation:
                l2-1 2s infinite,
                l2-2 2s infinite;
            }
            @keyframes l2-1 {
              0%   {background-size: 0    4px,4px 0   ,0    4px,4px 0   }
              12.5%{background-size: 100% 4px,4px 0   ,0    4px,4px 0   }
              25%  {background-size: 100% 4px,4px 100%,0    4px,4px 0   }
              37.5%{background-size: 100% 4px,4px 100%,100% 4px,4px 0   }
              45%,
              55%  {background-size: 100% 4px,4px 100%,100% 4px,4px 100%}
              62.5%{background-size: 0    4px,4px 100%,100% 4px,4px 100%}
              75%  {background-size: 0    4px,4px 0   ,100% 4px,4px 100%}
              87.5%{background-size: 0    4px,4px 0   ,0    4px,4px 100%}
              100% {background-size: 0    4px,4px 0   ,0    4px,4px 0   }
            }
            @keyframes l2-2 {
              0%,49.9%{background-position: 0 0   ,100% 0   ,100% 100%,0 100%}
              50%,100%{background-position: 100% 0,100% 100%,0    100%,0 0   }
            }
            </style>
            <div style="
              position:fixed;top:0;left:0;right:0;bottom:0;
              background:rgba(0,0,0,0.45);
              z-index:99999;
              display:flex;align-items:center;justify-content:center;
            ">
              <div style="
                background:#fff;border-radius:18px;
                padding:2.6rem 3.8rem;
                display:flex;flex-direction:column;align-items:center;gap:1rem;
                box-shadow:0 12px 48px rgba(0,0,0,0.18);
              ">
                <div class="loader"></div>
                <span style="color:#555;font-size:0.88rem;font-weight:600;letter-spacing:0.04em;">Processing...</span>
                <button onclick="(function(){var b=document.querySelector('[data-testid=stStatusWidget] button');if(b)b.click();})()"
                        style="background:none;border:none;padding:0;cursor:pointer;
                               color:#aaa;font-size:0.82rem;text-decoration:underline;">Stop</button>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _pil_img = st.session_state._cj_pending
        st.session_state._cj_loading = False
        st.session_state._cj_pending = None

        result, warn = make_listing(_pil_img)

        if warn:
            st.session_state.warn = warn
            st.rerun()
        else:
            st.session_state.original = _pil_img
            st.session_state.result   = result
            st.session_state.warn     = None
            st.rerun()

    elif _uploaded is not None:
        _pil_img = ImageOps.exif_transpose(Image.open(_uploaded))
        st.session_state._cj_pending = _pil_img
        st.session_state._cj_loading = True
        st.rerun()

# ── result screen ──────────────────────────────────────────────────────────
else:
    # ── Ratio selector ──────────────────────────────────────────────────
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

    _before = _crop_before(st.session_state.original, st.session_state.ratio)
    _after  = _crop_result(st.session_state.result,   st.session_state.ratio)

    _png_bytes = _to_png_bytes(_after)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='col-label'>Before</div>", unsafe_allow_html=True)
        st.image(_before, use_container_width=True)
    with col2:
        st.markdown("<div class='col-label'>After</div>", unsafe_allow_html=True)
        st.image(_after, use_container_width=True)

    # JS: clicking the image itself triggers the native Streamlit fullscreen button
    st.markdown(
        """<img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
             style="display:none"
             onload="(function(){
               function setup(){
                 document.querySelectorAll('[data-testid=stImage]').forEach(function(c){
                   var img=c.querySelector('img');
                   if(img&&!img.dataset.cjZoom){
                     img.dataset.cjZoom='1';
                     img.style.cursor='zoom-in';
                     img.addEventListener('click',function(){
                       var btn=c.querySelector('button');
                       if(btn)btn.click();
                     });
                   }
                 });
               }
               setTimeout(setup,150);
               setTimeout(setup,600);
             })();" />""",
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
            st.rerun()

