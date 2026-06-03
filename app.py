import base64
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

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
  .block-container {{ max-width: 880px; padding-top: 2.5rem; padding-bottom: 3rem; }}
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
  /* ── Navigation pill tabs ──────────────────────────────────────────── */
  /* Center the tab bar */
  div[data-testid="stTabs"] > div:first-child {{
    display: flex !important;
    justify-content: center !important;
    margin-bottom: 0.8rem !important;
  }}
  /* Pill container */
  div[data-testid="stTabBar"] {{
    background: white !important;
    border-radius: 999px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: none !important;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07) !important;
    overflow: visible !important;
    width: auto !important;
    min-width: 0 !important;
  }}
  /* Each tab button */
  button[data-baseweb="tab"] {{
    background: transparent !important;
    color: #888 !important;
    border-radius: 999px !important;
    padding: 9px 30px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    margin: 0 !important;
    transition: background 0.15s, color 0.15s !important;
    white-space: nowrap !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    background: {PRIMARY} !important;
    color: #006633 !important;
  }}
  button[data-baseweb="tab"][aria-selected="false"]:hover {{
    background: rgba(0,0,0,0.04) !important;
    color: #444 !important;
  }}
  /* Hide underline / border */
  div[data-baseweb="tab-highlight"],
  div[data-baseweb="tab-border"] {{
    display: none !important;
  }}
  /* Tab panel top padding */
  div[data-testid="stTabPanel"] {{
    padding-top: 1.2rem !important;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# ── rembg session (loaded once, cached for the lifetime of the server) ─────────

@st.cache_resource(show_spinner=False)
def _rembg_session():
    from rembg import new_session
    return new_session("u2net")


# ── image processing ───────────────────────────────────────────────────────────

def _extract_subject(pil_img: Image.Image):
    """
    Remove background with rembg/U2Net and return (rgb, alpha) as uint8 arrays.
    alpha is a smooth 0-255 channel — no hard thresholding done here.
    """
    from rembg import remove as rembg_remove
    rgba = rembg_remove(pil_img.convert("RGBA"), session=_rembg_session())
    arr  = np.array(rgba)
    return arr[:, :, :3], arr[:, :, 3]


def _tighten_alpha(alpha: np.ndarray) -> np.ndarray:
    """
    Post-process rembg alpha:
    1. Remove isolated noise specks (small open/close)
    2. Fill topologically enclosed holes (e.g. ring centre, frame window)
    3. Hard-zero pixels below confidence threshold
       – rembg assigns alpha 0-80 to background gaps between limbs;
         zeroing them forces those gaps to be fully transparent on the
         canvas (white) rather than showing a faint background colour.
    Soft product edges (alpha ≥ 80) are preserved for natural blending.
    """
    # ── 1. Binary mask + tiny speck cleanup ──────────────────────────────────
    _, binary = cv2.threshold(alpha, 15, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)

    # ── 2. Fill topologically enclosed holes ─────────────────────────────────
    inv = cv2.bitwise_not(binary)
    _, labels = cv2.connectedComponents(inv, connectivity=8)
    border_labels = set(np.concatenate([
        labels[0, :], labels[-1, :],
        labels[1:-1, 0], labels[1:-1, -1],
    ]).tolist())
    filled = binary.copy()
    for lbl in np.unique(labels):
        if lbl != 0 and lbl not in border_labels:
            filled[labels == lbl] = 255     # enclosed hole → opaque

    # ── 3. Re-apply alpha; fix holes; hard-zero low-confidence pixels ────────
    # Pixels inside enclosed holes had alpha=0 → set to 255 (product colour)
    # Pixels outside the mask → 0
    # Background-gap pixels (alpha 1-79) → 0 (transparent = canvas white)
    out = np.where(
        filled > 0,
        np.where(alpha > 0, alpha, np.uint8(255)),
        np.uint8(0),
    ).astype(np.uint8)
    out[out < 80] = 0   # kill semi-transparent background residue in gaps
    return out


def _color_guided_cleanup(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Second-pass cleanup: zero moderate-alpha pixels (80–210) whose colour
    closely matches the detected background colour.

    This catches concave "gap" areas (e.g. between arm and body) that rembg
    assigns surprisingly high alpha because they're surrounded by product
    pixels, yet their actual colour is the background.

    Skipped automatically when the background is near-white (mean ≥ 230) to
    avoid incorrectly removing white parts of the product.
    """
    h, w = rgb.shape[:2]
    b = max(10, min(30, h // 20, w // 20))

    # ── Estimate background colour from image border strips ──────────────────
    strips = [rgb[:b, :], rgb[-b:, :], rgb[:, :b], rgb[:, -b:]]
    border_px = np.concatenate([s.reshape(-1, 3) for s in strips]).astype(np.float32)
    if len(border_px) == 0:
        return alpha

    bg_color = np.median(border_px, axis=0)   # (3,) float32

    # Skip if background is near-white — colour distance too unreliable
    if float(np.mean(bg_color)) > 230:
        return alpha

    # ── Per-pixel L2 colour distance from background ─────────────────────────
    diff = rgb.astype(np.float32) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))   # (h, w)

    # Zero alpha where: uncertain zone (80–210) AND colour ≈ background
    out = alpha.copy()
    uncertain = (alpha >= 80) & (alpha <= 210)
    out[uncertain & (dist < 45)] = 0
    return out


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
    """
    rgb, alpha = _extract_subject(pil_img)
    alpha = _tighten_alpha(alpha)
    alpha = _color_guided_cleanup(rgb, alpha)   # zero gap pixels matching bg colour

    h, w = rgb.shape[:2]
    total = h * w

    fg_px = int(np.count_nonzero(alpha > 15))
    if fg_px < total * 0.01:
        return None, "Could not detect the item. Try a photo with a cleaner background."

    pts = cv2.findNonZero((alpha > 15).astype(np.uint8) * 255)
    if pts is None:
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
for _k in ("original", "result", "warn"):
    if _k not in st.session_state:
        st.session_state[_k] = None
if "ratio" not in st.session_state:
    st.session_state.ratio = "1:1"

# ── Navigation tabs ────────────────────────────────────────────────────────────
_tab_listing, _tab_social = st.tabs(["Item Listing", "Social Post"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Item Listing (CJ Formatter)
# ══════════════════════════════════════════════════════════════════════════════
with _tab_listing:
    st.markdown(
        f"<h2 style='color:{DARK};margin:0 0 1.4rem;font-size:1.85rem;font-weight:700;'>"
        "CJ Listing Formatter</h2>",
        unsafe_allow_html=True,
    )

    # ── upload screen ──────────────────────────────────────────────────────────
    if st.session_state.result is None:
        uploaded = st.file_uploader(
            "Drop photo here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        st.markdown(
            "<p style='text-align:center;color:#bbb;font-size:0.78rem;margin-top:-0.3rem;'>"
            "JPG · PNG · WEBP · up to 200 MB</p>",
            unsafe_allow_html=True,
        )

        if st.session_state.warn:
            st.markdown(
                f"<div class='warn-box'>&#9888; {st.session_state.warn}</div>",
                unsafe_allow_html=True,
            )

        if uploaded is not None:
            pil_img = ImageOps.exif_transpose(Image.open(uploaded))
            with st.spinner("Processing…"):
                result, warn = make_listing(pil_img)

            if warn:
                st.session_state.warn = warn
                st.rerun()
            else:
                st.session_state.original = pil_img
                st.session_state.result = result
                st.session_state.warn = None
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
        _b64 = base64.b64encode(_png_bytes).decode()
        _data_url = f"data:image/png;base64,{_b64}"

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<div class='col-label'>Before</div>", unsafe_allow_html=True)
            st.image(_before, use_container_width=True)
        with col2:
            _dl_name = f"cj_listing_{st.session_state.ratio.replace(':','x')}.png"
            st.markdown("<div class='col-label'>After</div>", unsafe_allow_html=True)
            st.markdown(
                f"""<div class="cj-after" id="cj-wrap">
                  <img id="cj-thumb" src="{_data_url}"
                       style="width:100%;border-radius:8px;display:block;cursor:zoom-in;" />
                  <button id="cj-expand" class="cj-expand-btn">
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="#555555">
                      <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                    </svg>
                  </button>
                </div>
                <div id="cj-lb">
                  <img id="cj-lb-img" src="{_data_url}" />
                  <a class="cj-lb-dl" id="cj-lb-dl" href="{_data_url}"
                     download="{_dl_name}">↓ Download</a>
                </div>
                <img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
                     style="display:none"
                     onload="(function(){{
                       var t=document.getElementById('cj-thumb'),
                           b=document.getElementById('cj-expand'),
                           lb=document.getElementById('cj-lb'),
                           im=document.getElementById('cj-lb-img'),
                           dl=document.getElementById('cj-lb-dl');
                       function op(){{lb.style.display='flex';}}
                       function cl(){{lb.style.display='none';}}
                       if(t)t.addEventListener('click',op);
                       if(b)b.addEventListener('click',function(e){{e.stopPropagation();op();}});
                       if(lb)lb.addEventListener('click',function(e){{if(e.target===lb)cl();}});
                       if(im)im.addEventListener('click',function(e){{e.stopPropagation();}});
                       if(dl)dl.addEventListener('click',function(e){{e.stopPropagation();}});
                     }})();" />""",
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · Social Post (IAAC Identity Tool)
# ══════════════════════════════════════════════════════════════════════════════
with _tab_social:
    # The IAAC tool is a WebGL/JS app served from Streamlit's static file
    # serving at /app/static/iaac/index.html.  We use JS to set the iframe src
    # dynamically so the URL is correct both locally and on Streamlit Cloud.
    st.markdown(
        """
        <iframe id="iaac-frame"
          style="width:100%;height:900px;border:none;border-radius:8px;display:block;"
          allow="camera; microphone; clipboard-write; downloads"
          allowfullscreen></iframe>
        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
             style="display:none"
             onload="(function(){
               var f = document.getElementById('iaac-frame');
               if(f && !f.src) {
                 f.src = window.location.origin + '/app/static/iaac/index.html';
               }
             })();" />
        """,
        unsafe_allow_html=True,
    )
