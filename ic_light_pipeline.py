"""
IC-Light relighting pipeline — foreground-conditioned model (fc variant).

Applies consistent left-side studio lighting to an already-extracted product image.

First use downloads:
  ~350 MB  IC-Light delta weights  (huggingface.co/lllyasviel/ic-light)
  ~5.1 GB  SD 1.5 base model       (stablediffusionapi/realistic-vision-v51)
Both are cached in ~/.cache/huggingface/ and reused on subsequent runs.

Expected performance per image:
  CUDA (NVIDIA)        ~5-10 s   (float16)
  MPS  (Apple Silicon) ~60-120 s (float32)
  CPU                  ~10+ min  (float32, not recommended)
"""

import math
import os
import logging
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file

log = logging.getLogger("cj_listing.ic_light")

# ── model / weight locations ───────────────────────────────────────────────────
_SD15_REPO  = "stablediffusionapi/realistic-vision-v51"
_IC_URL     = "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors"
_IC_PATH    = os.path.join(os.path.expanduser("~"), ".cache", "ic_light", "iclight_sd15_fc.safetensors")

# ── fixed relighting parameters (left-side studio light) ──────────────────────
PROMPT          = "soft studio lighting from the left, professional product photography"
ADDED_PROMPT    = "best quality, sharp, clean"
NEG_PROMPT      = "lowres, cropped, worst quality, blurry, noisy, overexposed"
SEED            = 42       # fixed seed → consistent results across batches
STEPS           = 20
CFG             = 2.0      # IC-Light works at low CFG; higher values over-lighten
LOWRES_DENOISE  = 0.9
HIGHRES_SCALE   = 1.5      # 512 → 768 px
HIGHRES_DENOISE = 0.5
BASE_SIZE       = 512


# ── device / dtype ────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unet_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def _vae_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


# ── model loading ─────────────────────────────────────────────────────────────

def load_models() -> dict:
    """
    Load and return the IC-Light pipeline components as a plain dict.
    Call this once; wrap with @st.cache_resource in app.py to reuse across
    Streamlit reruns without reloading.
    """
    import safetensors.torch as sf
    from diffusers import (
        AutoencoderKL, UNet2DConditionModel,
        StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
        DPMSolverMultistepScheduler,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTokenizer

    device    = _device()
    u_dtype   = _unet_dtype(device)
    v_dtype   = _vae_dtype(device)
    log.info("IC-Light loading  device=%s  unet=%s  vae=%s", device, u_dtype, v_dtype)

    tokenizer    = CLIPTokenizer.from_pretrained(_SD15_REPO, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(_SD15_REPO, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(_SD15_REPO, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(_SD15_REPO, subfolder="unet")

    # ── Widen UNet input: 4 → 8 channels for foreground conditioning ──────────
    with torch.no_grad():
        new_in = torch.nn.Conv2d(
            8, unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_in.weight.zero_()
        new_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_in.bias = unet.conv_in.bias
        unet.conv_in = new_in

    # ── Forward hook: inject fg-conditioning latent at every forward pass ─────
    _orig_fwd = unet.forward

    def _hooked_fwd(sample, timestep, encoder_hidden_states, **kw):
        cond = kw["cross_attention_kwargs"]["concat_conds"].to(sample)
        cond = torch.cat([cond] * (sample.shape[0] // cond.shape[0]), dim=0)
        kw["cross_attention_kwargs"] = {}
        return _orig_fwd(torch.cat([sample, cond], dim=1), timestep, encoder_hidden_states, **kw)

    unet.forward = _hooked_fwd

    # ── Download IC-Light delta weights and merge into UNet ───────────────────
    os.makedirs(os.path.dirname(_IC_PATH), exist_ok=True)
    if not os.path.exists(_IC_PATH):
        log.info("Downloading IC-Light weights → %s  (~350 MB)", _IC_PATH)
        download_url_to_file(_IC_URL, _IC_PATH)

    offset = sf.load_file(_IC_PATH)
    origin = unet.state_dict()
    unet.load_state_dict({k: origin[k] + offset[k] for k in origin}, strict=True)
    del offset, origin

    # ── Move to device ────────────────────────────────────────────────────────
    text_encoder = text_encoder.to(device=device, dtype=u_dtype)
    vae          = vae.to(device=device, dtype=v_dtype)
    unet         = unet.to(device=device, dtype=u_dtype)

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1,
    )
    base_kwargs = dict(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler,
        safety_checker=None, requires_safety_checker=False,
        feature_extractor=None, image_encoder=None,
    )
    t2i = StableDiffusionPipeline(**base_kwargs)
    i2i = StableDiffusionImg2ImgPipeline(**base_kwargs)

    log.info("IC-Light models ready")
    return dict(device=device, vae=vae, unet=unet,
                text_encoder=text_encoder, tokenizer=tokenizer,
                t2i=t2i, i2i=i2i)


# ── tensor helpers ────────────────────────────────────────────────────────────

def _np2pt(imgs: list) -> torch.Tensor:
    """uint8 (H,W,3) list → float32 (N,3,H,W) tensor in [-1, 1]."""
    return torch.from_numpy(np.stack(imgs, 0)).float() / 127.0 - 1.0


def _pt2np(t: torch.Tensor) -> list:
    """(N,3,H,W) tensor → list of uint8 (H,W,3) arrays."""
    return [
        (x.movedim(0, -1) * 127.5 + 127.5)
        .detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        for x in t
    ]


def _encode_prompt(tokenizer, text_encoder, device, text: str) -> torch.Tensor:
    max_len   = tokenizer.model_max_length
    chunk_len = max_len - 2
    bos, eos  = tokenizer.bos_token_id, tokenizer.eos_token_id

    raw = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [
        ([bos] + raw[i:i + chunk_len] + [eos])[:max_len]
        for i in range(0, max(len(raw), 1), chunk_len)
    ]
    chunks = [c + [eos] * (max_len - len(c)) for c in chunks]
    ids = torch.tensor(chunks, dtype=torch.int64, device=device)
    return text_encoder(ids).last_hidden_state


def _resize_center_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    pil   = Image.fromarray(img)
    scale = max(w / pil.width, h / pil.height)
    rw    = int(round(pil.width  * scale))
    rh    = int(round(pil.height * scale))
    pil   = pil.resize((rw, rh), Image.LANCZOS)
    l     = (rw - w) // 2
    t     = (rh - h) // 2
    return np.array(pil.crop((l, t, l + w, t + h)))


def _left_gradient(h: int, w: int) -> np.ndarray:
    """Bright-on-left → dark-on-right gradient, used as IC-Light initial latent."""
    grad = np.linspace(255, 0, w)
    tile = np.tile(grad, (h, 1))
    return np.stack([tile] * 3, axis=-1).astype(np.uint8)


# ── public API ────────────────────────────────────────────────────────────────

def prepare_fg(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Blend the product onto a neutral gray (127) background.

    IC-Light expects the foreground in this format — gray where transparent,
    product colour where opaque — matching the output of its own RMBG pre-processing.
    """
    a    = alpha.astype(np.float32) / 255.0
    gray = np.full_like(rgb, 127, dtype=np.float32)
    fg   = gray + (rgb.astype(np.float32) - gray) * a[:, :, np.newaxis]
    return np.clip(fg, 0, 255).astype(np.uint8)


@torch.inference_mode()
def relight_left(
    fg_np:    np.ndarray,
    alpha_np: np.ndarray,
    models:   dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Relight the product with consistent left-side studio lighting.

    Parameters
    ----------
    fg_np    : (H, W, 3) uint8  — product blended onto gray127 (from prepare_fg)
    alpha_np : (H, W)    uint8  — alpha mask from rembg (same spatial size as fg_np)
    models   : dict returned by load_models()

    Returns
    -------
    relit_rgb    : (H', W', 3) uint8  — relit product pixels
    aligned_alpha: (H', W')    uint8  — alpha resized + center-cropped to match relit_rgb
    """
    device  = models["device"]
    vae     = models["vae"]
    i2i     = models["i2i"]
    tok     = models["tokenizer"]
    tenc    = models["text_encoder"]

    W = H = BASE_SIZE
    rng = torch.Generator(device=device).manual_seed(SEED)

    # ── Encode text prompts ───────────────────────────────────────────────────
    pos = PROMPT + ", " + ADDED_PROMPT
    c   = _encode_prompt(tok, tenc, device, pos)
    uc  = _encode_prompt(tok, tenc, device, NEG_PROMPT)

    max_chunk = max(len(c), len(uc))
    c  = torch.cat([c]  * math.ceil(max_chunk / len(c)),  dim=0)[:max_chunk]
    uc = torch.cat([uc] * math.ceil(max_chunk / len(uc)), dim=0)[:max_chunk]
    c  = torch.cat([p[None] for p in c],  dim=1)
    uc = torch.cat([p[None] for p in uc], dim=1)

    # ── Encode foreground conditioning latent ─────────────────────────────────
    fg_512    = _resize_center_crop(fg_np, W, H)
    fg_latent = _np2pt([fg_512]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(fg_latent).latent_dist.mode() * vae.config.scaling_factor

    # ── Left-gradient background → initial latent ─────────────────────────────
    bg_np  = _left_gradient(H, W)
    bg_lat = _np2pt([bg_np]).to(device=vae.device, dtype=vae.dtype)
    bg_lat = vae.encode(bg_lat).latent_dist.mode() * vae.config.scaling_factor

    # ── Low-res pass (512 px) conditioned on gradient background ─────────────
    latents = i2i(
        image=bg_lat,
        strength=LOWRES_DENOISE,
        prompt_embeds=c,
        negative_prompt_embeds=uc,
        width=W, height=H,
        num_inference_steps=int(round(STEPS / LOWRES_DENOISE)),
        num_images_per_prompt=1,
        generator=rng,
        output_type="latent",
        guidance_scale=CFG,
        cross_attention_kwargs={"concat_conds": concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    # ── Decode → upscale → re-encode ─────────────────────────────────────────
    pixels = _pt2np(vae.decode(latents).sample)
    WW = int(round(W * HIGHRES_SCALE / 64) * 64)
    HH = int(round(H * HIGHRES_SCALE / 64) * 64)
    pixels = [np.array(Image.fromarray(p).resize((WW, HH), Image.LANCZOS)) for p in pixels]

    latents = _np2pt(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(latents).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=models["unet"].device, dtype=models["unet"].dtype)

    # ── Re-encode fg at high-res size ─────────────────────────────────────────
    fg_hires    = _resize_center_crop(fg_np, WW, HH)
    cond_hires  = _np2pt([fg_hires]).to(device=vae.device, dtype=vae.dtype)
    cond_hires  = vae.encode(cond_hires).latent_dist.mode() * vae.config.scaling_factor

    # ── High-res refinement pass ──────────────────────────────────────────────
    latents = i2i(
        image=latents,
        strength=HIGHRES_DENOISE,
        prompt_embeds=c,
        negative_prompt_embeds=uc,
        width=WW, height=HH,
        num_inference_steps=int(round(STEPS / HIGHRES_DENOISE)),
        num_images_per_prompt=1,
        generator=rng,
        output_type="latent",
        guidance_scale=CFG,
        cross_attention_kwargs={"concat_conds": cond_hires},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    relit_rgb = _pt2np(vae.decode(latents).sample)[0]  # (HH, WW, 3)

    # ── Align alpha: same resize + center-crop as fg_np went through ──────────
    # alpha_np is (H_orig, W_orig) — apply identical spatial transform
    alpha_pil      = Image.fromarray(alpha_np)
    orig_h, orig_w = alpha_np.shape[:2]
    scale          = max(WW / orig_w, HH / orig_h)
    rw             = int(round(orig_w * scale))
    rh             = int(round(orig_h * scale))
    alpha_pil      = alpha_pil.resize((rw, rh), Image.LANCZOS)
    l              = (rw - WW) // 2
    t              = (rh - HH) // 2
    aligned_alpha  = np.array(alpha_pil.crop((l, t, l + WW, t + HH)))

    log.info(
        "IC-Light done  output=%dx%d  device=%s",
        relit_rgb.shape[1], relit_rgb.shape[0], device,
    )
    return relit_rgb, aligned_alpha
