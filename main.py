#!/usr/bin/env python3
"""
Batch JPEG optimizer with SSIM target; robust RAW/TIFF loading; metadata/sidecars preserved;
OpenCV BRISQUE optional; multiprocessing; resume; flat-mode de-dup; file-type filtering;
smart 16-bit TIFF percentile scaling; DNG color improvements (camera WB).

Usage examples:
  python main.py "/path/to/Camera Roll" --ssim 0.95 --workers 8 --resume
  python main.py "/path/to/Camera Roll" --types tif,tiff,dng --tiff-smart16 --tiff-smart16-pct 0.5,99.5
  python main.py "/path/to/Camera Roll" --flat --workers 6 \
      --brisque-model ~/models/brisque/BRISQUE_model_live.yml \
      --brisque-range ~/models/brisque/BRISQUE_range_live.yml
"""

import argparse
import io
import os
import shutil
import subprocess
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, Set, List, Dict, Any
import json
from dataclasses import dataclass

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

import numpy as np
from PIL import Image, ImageFile
try:
    from PIL import ImageCms
    HAVE_IMAGECMS = True
except Exception:
    HAVE_IMAGECMS = False
from skimage.metrics import structural_similarity as ssim

import time
try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

# Allow truncated reads for TIFF/JPEG
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional: better HEIC/HEIF support via Pillow plugin
try:
    import pillow_heif  # pip install pillow-heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# Optional deps
try:
    import rawpy
    HAVE_RAWPY = True
except Exception:
    HAVE_RAWPY = False

# OpenCV (for BRISQUE)
try:
    import cv2
    HAVE_CV2 = True
    HAVE_CV2_QUALITY = hasattr(cv2, "quality")
    HAVE_CV2_BRISQUE = HAVE_CV2_QUALITY and hasattr(cv2.quality, "QualityBRISQUE_create")
except Exception:
    HAVE_CV2 = False
    HAVE_CV2_QUALITY = False
    HAVE_CV2_BRISQUE = False

# ----------------------------
# Config
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}
RAW_EXTS: Set[str] = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".srw", ".pef"}
SIDECAR_EXTS: Set[str] = {".xmp", ".xml", ".json"}  # extend as needed

# Default BRISQUE model paths (next to this script by default)
DEFAULT_BRISQUE_MODEL = str(SCRIPT_DIR / "brisque_model_live.yml")
DEFAULT_BRISQUE_RANGE = str(SCRIPT_DIR / "brisque_range_live.yml")

# Smart 16-bit TIFF scaling controls (set from CLI at startup)
TIFF_SMART16: bool = False
TIFF_SMART16_PCTS: Tuple[float, float] = (0.5, 99.5)  # (low, high)
TIFF_SMART16_PERCHANNEL: bool = False  # default to per-channel stretch; set False to use global curve
# Exposure/Gamma globals for 16-bit TIFF mapping (set in __main__ from CLI)
TIFF_GAMMA: Optional[float] = None
TIFF_EXPOSURE_EV: float = 0.0


# Worker process initializer - sets globals for multiprocessing
def _init_worker_globals(settings_dict):
    """Initialize global variables in worker processes."""
    global TIFF_SMART16, TIFF_SMART16_PCTS, TIFF_SMART16_PERCHANNEL
    global SMART16_DOWNSAMPLE, SSIM_DOWNSAMPLE, SSIM_LUMA_ONLY, SEARCH_OPTIMIZE
    global TIFF_GAMMA, TIFF_EXPOSURE_EV, TIFF_FLOAT_TONEMAP
    global AUTO_EV_MODE, AUTO_EV_MID, AUTO_EV_MID_PCT, AUTO_EV_HI_PCT, AUTO_EV_HI_CAP
    global AUTO_EV_DOWNSAMPLE, AUTO_EV_BOUNDS, AUTO_EV_ITERS
    global BLACKPOINT_PCT, WHITEPOINT_PCT, SHADOW_LIFT, CONTRAST_STRENGTH, SATURATION

    for key, value in settings_dict.items():
        globals()[key] = value



def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def has_exiftool() -> bool:
    try:
        subprocess.run(["exiftool", "-ver"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

EXIFTOOL_OK = has_exiftool()

# ----------------------------
# Sidecars
# ----------------------------
def copy_sidecars(src_path: Path, dst_path_without_ext: Path):
    """Copy sidecar files (e.g., .xmp) to the mirrored output path."""
    stem = src_path.with_suffix("")
    for ext in SIDECAR_EXTS:
        sidecar = stem.with_suffix(ext)
        if sidecar.exists():
            dst_sidecar = dst_path_without_ext.with_suffix(ext)
            ensure_dir(dst_sidecar)
            shutil.copy2(sidecar, dst_sidecar)

# ----------------------------
# Robust loading
# ----------------------------

@dataclass
class MapStats:
    loader: str = ""
    dtype: str = ""
    shape: tuple = ()
    bits_per_sample: str = ""
    sample_format: str = ""      # 1=uint, 2=int, 3=float, etc.
    photometric: str = ""        # RGB, MINISBLACK, MINISWHITE, etc.
    icc_applied: bool = False
    ev_applied: float = 0.0
    gamma_applied: float | None = None
    tonemap: str = "none"
    p_lo: float = 0.0
    p_hi: float = 100.0
    per_channel: bool = False
    src_min: float = 0.0
    src_max: float = 0.0
    src_lo_val: list | float | None = None
    src_hi_val: list | float | None = None
    out_min: float = 0.0
    out_max: float = 1.0


def _percentile_stretch(arrf: np.ndarray, lo: float, hi: float, per_channel: bool) -> tuple[np.ndarray, list | float, list | float]:
    """Return arr01 in [0,1] and the lo/hi values used (per-channel or global)."""
    ds = int(globals().get("SMART16_DOWNSAMPLE", 1))
    a = arrf[::ds, ::ds] if arrf.ndim == 2 else arrf[::ds, ::ds, :]
    if arrf.ndim == 2:  # grayscale
        lo_v, hi_v = np.nanpercentile(a, (lo, hi))
        if hi_v <= lo_v:
            arr01 = np.clip(arrf, 0.0, 1.0)
        else:
            arr01 = np.clip((arrf - lo_v) / (hi_v - lo_v), 0.0, 1.0)
        return arr01[..., None], float(lo_v), float(hi_v)

    # color
    h, w, c = arrf.shape
    c = min(c, 3)
    out = np.empty((h, w, 3), np.float32)
    if per_channel:
        lo_vals, hi_vals = [], []
        for ch in range(c):
            lo_v, hi_v = np.nanpercentile(a[..., ch], (lo, hi))
            lo_vals.append(float(lo_v)); hi_vals.append(float(hi_v))
            if hi_v <= lo_v:
                out[..., ch] = np.clip(arrf[..., ch], 0.0, 1.0)
            else:
                out[..., ch] = np.clip((arrf[..., ch] - lo_v) / (hi_v - lo_v), 0.0, 1.0)
        # duplicate last channel if c<3
        for ch in range(c, 3):
            out[..., ch] = out[..., c-1]
        return out, lo_vals, hi_vals
    else:
        lo_v, hi_v = np.nanpercentile(a[..., :c], (lo, hi))
        if hi_v <= lo_v:
            out[..., :c] = np.clip(arrf[..., :c], 0.0, 1.0)
        else:
            out[..., :c] = np.clip((arrf[..., :c] - lo_v) / (hi_v - lo_v), 0.0, 1.0)
        for ch in range(c, 3):
            out[..., ch] = out[..., c-1]
        return out, float(lo_v), float(hi_v)


def _tonemap(arr: np.ndarray, mode: str) -> np.ndarray:
    """arr is linear and non-negative; return linear [0, +) mapped into [0,1]."""
    if mode == "none":
        return arr
    if mode == "reinhard":
        # classic Reinhard: x/(1+x); safe per-channel
        return arr / (1.0 + arr)
    if mode == "aces":
        # ACES (Hable/ACES approximation) – widely used filmic curve
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((arr*(a*arr + b)) / (arr*(c*arr + d) + e), 0.0, 1.0)
    return arr

def _lin_luma(a: np.ndarray) -> np.ndarray:
    """Scene-linear luma; a must be (...,3) in [0,1]."""
    return 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]

# ----- Auto-EV helpers -----

def _subsample(arr: np.ndarray, step: int) -> np.ndarray:
    return arr[::step, ::step, :] if arr.ndim == 3 else arr[::step, ::step]


def _downsample_arr(a: np.ndarray, step: int) -> np.ndarray:
    return a[::step, ::step] if step > 1 else a


def _rgb_to_luma8(a: np.ndarray) -> np.ndarray:
    # expects uint8 RGB
    return (0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]).astype(np.uint8)


def _bisect_ev(f, target: float, lo: float, hi: float, iters: int) -> float:
    """Find EV where f(EV) ~= target (f is monotonic in EV)."""
    for _ in range(max(4, iters)):
        mid = 0.5*(lo+hi)
        val = f(mid)
        if np.isnan(val):
            hi = mid  # shrink interval
            continue
        if val < target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


def _solve_auto_ev(arr01_lin: np.ndarray, tonemap_mode: str,
                   mid_target: float, mid_pct: float,
                   hi_pct: float | None, hi_cap: float | None,
                   ds: int, bounds: tuple[float,float], iters: int) -> dict:
    """
    arr01_lin: percentile-stretched linear RGB in [0,1] BEFORE EV.
    Returns {'ev_mid':..., 'ev_hi':..., 'ev_final':..., 'mid_val':..., 'hi_val':...}
    """
    # downsample to speed up
    a = _subsample(arr01_lin, ds)
    if a.ndim != 3 or a.shape[-1] < 3:
        return {"ev_mid": 0.0, "ev_hi": None, "ev_final": 0.0, "mid_val": float("nan"), "hi_val": float("nan")}
    # evaluate pipeline for a given EV on the sample
    def eval_ev(ev: float) -> tuple[float, float]:
        arr_ev = np.clip(a * (2.0**ev), 0.0, None)
        arr_tm = _tonemap(arr_ev, tonemap_mode)   # still linear
        Y = _lin_luma(arr_tm)
        mid_val = float(np.nanpercentile(Y, mid_pct))
        hi_val = float(np.nanpercentile(Y, hi_pct)) if hi_pct is not None else float("nan")
        return mid_val, hi_val

    # 1) solve for mid target
    def f_mid(ev: float) -> float:
        m, _ = eval_ev(ev); return m
    ev_mid = _bisect_ev(f_mid, mid_target, bounds[0], bounds[1], iters)
    mid_val, hi_val_at_mid = eval_ev(ev_mid)

    ev_hi = None
    if hi_pct is not None and hi_cap is not None and np.isfinite(hi_val_at_mid):
        # 2) solve highlight cap
        def f_hi(ev: float) -> float:
            _, h = eval_ev(ev); return h
        ev_hi = _bisect_ev(f_hi, hi_cap, bounds[0], bounds[1], iters)
        # take the safer (darker) EV
        ev_final = min(ev_mid, ev_hi)
        _, hi_val = eval_ev(ev_final)
        return {"ev_mid": ev_mid, "ev_hi": ev_hi, "ev_final": ev_final,
                "mid_val": mid_val, "hi_val": hi_val}
    else:
        return {"ev_mid": ev_mid, "ev_hi": None, "ev_final": ev_mid,
                "mid_val": mid_val, "hi_val": float("nan")}


def _apply_post_gamma_shaping(arr_disp: np.ndarray, dbg=None) -> np.ndarray:
    """arr_disp in [0,1] (after gamma). Apply black/white point, contrast S-curve, saturation."""
    # Black/white point remap on luminance percentiles
    bp = globals().get("BLACKPOINT_PCT", None)
    wp = globals().get("WHITEPOINT_PCT", None)
    if bp is not None or wp is not None:
        Y = _lin_luma(arr_disp)
        if bp is None: bp = 0.0
        if wp is None: wp = 100.0
        bval = float(np.nanpercentile(Y, bp))
        wval = float(np.nanpercentile(Y, wp))
        if wval > bval + 1e-6:
            arr_disp = np.clip((arr_disp - bval) / (wval - bval), 0.0, 1.0)
        if dbg is not None:
            dbg.__dict__.update(bp_pct=bp, wp_pct=wp, bp_val=bval, wp_val=wval)

    # Shadow lift (brightens dark areas without affecting highlights)
    shadow_lift = float(globals().get("SHADOW_LIFT", 0.0) or 0.0)
    if abs(shadow_lift) > 1e-6:
        Y = _lin_luma(arr_disp)
        # Smooth mask: strong in shadows, fades to zero in highlights
        # Using (1-Y)^2 gives a smooth quadratic falloff
        mask = np.power(np.clip(1.0 - Y, 0.0, 1.0), 2.0)
        # Add lift proportional to how dark the pixel is
        arr_disp = np.clip(arr_disp + shadow_lift * mask[..., None], 0.0, 1.0)
        if dbg is not None:
            dbg.__dict__.update(shadow_lift=shadow_lift)

    # Contrast S-curve (sigmoid around 0.5)
    c = float(globals().get("CONTRAST_STRENGTH", 0.0) or 0.0)
    if abs(c) > 1e-6:
        # gentle: gain ~ 2..4 as c goes 0..0.5
        gain = 2.0 + 4.0*c*2.0
        arr_disp = 0.5 + np.tanh((arr_disp - 0.5) * gain) * 0.5

    # Saturation in display space
    sat = float(globals().get("SATURATION", 1.0))
    if abs(sat - 1.0) > 1e-6:
        Y = _lin_luma(arr_disp)[..., None]
        arr_disp = np.clip(Y + (arr_disp - Y) * sat, 0.0, 1.0)

    return arr_disp


def _apply_exposure_and_gamma_01(arr01: np.ndarray, ev: float, gamma: Optional[float]) -> np.ndarray:
    """
    arr01: float32 in [0,1]. Apply exposure (EV) then gamma (to sRGB-like display).
    """
    if ev and ev != 0.0:
        arr01 = arr01 * (2.0 ** ev)
    arr01 = np.clip(arr01, 0.0, 1.0)
    if gamma and gamma > 0:
        arr01 = np.power(arr01, 1.0 / gamma)  # linear -> display
    return arr01

def _to_uint8_rgb(arr: np.ndarray, dbg: MapStats | None = None) -> np.ndarray:
    """
    Map uint16/float TIFF data to 8-bit RGB using:
      - percentile stretch (applies to float too)
      - optional auto-EV anchor
      - EV, optional tonemap, gamma
    """
    ev = globals().get("TIFF_EXPOSURE_EV", 0.0)
    gamma = globals().get("TIFF_GAMMA", None)
    tonemap_mode = globals().get("TIFF_FLOAT_TONEMAP", "none")
    use_smart = bool(globals().get("TIFF_SMART16", False))
    per_channel = bool(globals().get("TIFF_SMART16_PERCHANNEL", False))
    lo, hi = globals().get("TIFF_SMART16_PCTS", (0.5, 99.5))
    auto_mid = globals().get("AUTO_EV_MID", None)
    auto_pct = float(globals().get("AUTO_EV_PCT", 50.0))

    def _finish(arr_disp: np.ndarray):
        # Apply optional post-gamma shaping (black/white point, contrast, saturation)
        arr_disp = _apply_post_gamma_shaping(arr_disp, dbg)
        if dbg:
            dbg.out_min = float(arr_disp.min()); dbg.out_max = float(arr_disp.max())
        if arr_disp.shape[-1] == 1:
            arr_disp = np.repeat(arr_disp, 3, axis=-1)
        return (arr_disp * 255.0 + 0.5).astype(np.uint8)

    # -------- uint16 --------
    if arr.dtype == np.uint16:
        arrf = arr.astype(np.float32)
        if use_smart:
            arr01, lo_v, hi_v = _percentile_stretch(arrf, lo, hi, per_channel)
        else:
            arr01 = arrf / 65535.0
            if arr01.ndim == 2: arr01 = arr01[..., None]
            lo_v, hi_v = 0.0, 65535.0

        # ----- Auto EV (per-image) -----
        auto_mode = globals().get("AUTO_EV_MODE", "off")
        manual_ev = globals().get("TIFF_EXPOSURE_EV", 0.0) or 0.0
        if auto_mode != "off" and arr01.ndim == 3 and arr01.shape[-1] >= 3:
            # solve EV to hit mid target and guard highlights
            res = _solve_auto_ev(
                arr01_lin=arr01,
                tonemap_mode=globals().get("TIFF_FLOAT_TONEMAP", "none"),
                mid_target=float(globals().get("AUTO_EV_MID", 0.18)),
                mid_pct=float(globals().get("AUTO_EV_MID_PCT", 50.0)),
                hi_pct=float(globals().get("AUTO_EV_HI_PCT", 99.0)) if auto_mode == "mid_guard" else None,
                hi_cap=float(globals().get("AUTO_EV_HI_CAP", 0.90)) if auto_mode == "mid_guard" else None,
                ds=int(globals().get("AUTO_EV_DOWNSAMPLE", 8)),
                bounds=globals().get("AUTO_EV_BOUNDS", (-4.0, 6.0)),
                iters=int(globals().get("AUTO_EV_ITERS", 16)),
            )
            ev_auto = float(res["ev_final"])
            # apply auto + manual offset
            arr01 = arr01 * (2.0 ** (ev_auto + manual_ev))
            if dbg is not None:
                dbg.__dict__.update(
                    auto_ev_mode=auto_mode,
                    auto_ev_mid=res.get("ev_mid"),
                    auto_ev_hi=res.get("ev_hi"),
                    auto_ev_final=ev_auto + manual_ev,
                    auto_mid_target=globals().get("AUTO_EV_MID", 0.18),
                    auto_mid_pct=globals().get("AUTO_EV_MID_PCT", 50.0),
                    auto_hi_pct=globals().get("AUTO_EV_HI_PCT", 99.0),
                    auto_hi_cap=globals().get("AUTO_EV_HI_CAP", 0.90),
                )
        else:
            # apply only manual EV here if auto is off
            if manual_ev:
                arr01 = arr01 * (2.0 ** manual_ev)
        # prevent double EV application in subsequent steps
        ev = 0.0

        # EV, tonemap, gamma
        if ev: arr01 = arr01 * (2.0 ** ev)
        arr_lin = np.clip(arr01, 0.0, None)
        arr_tm = _tonemap(arr_lin, tonemap_mode)
        arr_tm = np.clip(arr_tm, 0.0, 1.0)
        arr_disp = np.power(arr_tm, 1.0 / gamma) if (gamma and gamma > 0) else arr_tm

        if dbg:
            dbg.src_min = float(arrf.min()); dbg.src_max = float(arrf.max())
            dbg.p_lo = lo; dbg.p_hi = hi; dbg.per_channel = per_channel
            dbg.src_lo_val = lo_v; dbg.src_hi_val = hi_v
            dbg.ev_applied = ev; dbg.gamma_applied = gamma; dbg.tonemap = tonemap_mode
            dbg.__dict__.update(linY_p50_pre=float(np.nanpercentile(_lin_luma(arr01),50)),
                                linY_p50_post=float(np.nanpercentile(_lin_luma(arr_tm),50)))
        return _finish(arr_disp)

    # -------- float32/64 --------
    if np.issubdtype(arr.dtype, np.floating):
        arrf = arr.astype(np.float32)
        if use_smart:
            arr01, lo_v, hi_v = _percentile_stretch(arrf, lo, hi, per_channel)
        else:
            lo_v, hi_v = float(np.nanmin(arrf)), float(np.nanpercentile(arrf, 99.9))
            rng = max(1e-6, hi_v - lo_v)
            arr01 = np.clip((arrf - lo_v) / rng, 0.0, 1.0)
            if arr01.ndim == 2: arr01 = arr01[..., None]

        # ----- Auto EV (per-image) -----
        auto_mode = globals().get("AUTO_EV_MODE", "off")
        manual_ev = globals().get("TIFF_EXPOSURE_EV", 0.0) or 0.0
        if auto_mode != "off" and arr01.ndim == 3 and arr01.shape[-1] >= 3:
            # solve EV to hit mid target and guard highlights
            res = _solve_auto_ev(
                arr01_lin=arr01,
                tonemap_mode=globals().get("TIFF_FLOAT_TONEMAP", "none"),
                mid_target=float(globals().get("AUTO_EV_MID", 0.18)),
                mid_pct=float(globals().get("AUTO_EV_MID_PCT", 50.0)),
                hi_pct=float(globals().get("AUTO_EV_HI_PCT", 99.0)) if auto_mode == "mid_guard" else None,
                hi_cap=float(globals().get("AUTO_EV_HI_CAP", 0.90)) if auto_mode == "mid_guard" else None,
                ds=int(globals().get("AUTO_EV_DOWNSAMPLE", 8)),
                bounds=globals().get("AUTO_EV_BOUNDS", (-4.0, 6.0)),
                iters=int(globals().get("AUTO_EV_ITERS", 16)),
            )
            ev_auto = float(res["ev_final"])
            # apply auto + manual offset
            arr01 = arr01 * (2.0 ** (ev_auto + manual_ev))
            if dbg is not None:
                dbg.__dict__.update(
                    auto_ev_mode=auto_mode,
                    auto_ev_mid=res.get("ev_mid"),
                    auto_ev_hi=res.get("ev_hi"),
                    auto_ev_final=ev_auto + manual_ev,
                    auto_mid_target=globals().get("AUTO_EV_MID", 0.18),
                    auto_mid_pct=globals().get("AUTO_EV_MID_PCT", 50.0),
                    auto_hi_pct=globals().get("AUTO_EV_HI_PCT", 99.0),
                    auto_hi_cap=globals().get("AUTO_EV_HI_CAP", 0.90),
                )
        else:
            # apply only manual EV here if auto is off
            if manual_ev:
                arr01 = arr01 * (2.0 ** manual_ev)
        # prevent double EV application in subsequent steps
        ev = 0.0

        if ev: arr01 = arr01 * (2.0 ** ev)
        arr_lin = np.clip(arr01, 0.0, None)
        arr_tm = _tonemap(arr_lin, tonemap_mode)
        arr_tm = np.clip(arr_tm, 0.0, 1.0)
        arr_disp = np.power(arr_tm, 1.0 / gamma) if (gamma and gamma > 0) else arr_tm

        if dbg:
            dbg.src_min = float(np.nanmin(arrf)); dbg.src_max = float(np.nanmax(arrf))
            dbg.p_lo = lo; dbg.p_hi = hi; dbg.per_channel = per_channel
            dbg.src_lo_val = lo_v; dbg.src_hi_val = hi_v
            dbg.ev_applied = ev; dbg.gamma_applied = gamma; dbg.tonemap = tonemap_mode
            dbg.__dict__.update(linY_p50_pre=float(np.nanpercentile(_lin_luma(arr01),50)),
                                linY_p50_post=float(np.nanpercentile(_lin_luma(arr_tm),50)))
        return _finish(arr_disp)

    # -------- already 8-bit etc. --------
    arr8 = arr.astype(np.uint8, copy=False)
    if arr8.ndim == 2: return np.stack([arr8, arr8, arr8], axis=-1)
    if arr8.ndim == 3 and arr8.shape[-1] >= 3: return arr8[:, :, :3]
    return np.repeat(arr8[..., None], 3, axis=-1)

def _open_tiff_robust(path: Path, tiff_apply_icc: bool, tiff_gamma: Optional[float], tiff_exposure_ev: float,
                      tiff_reader: str = "auto") -> Image.Image:
    dbg = MapStats()
    dbg.ev_applied = tiff_exposure_ev
    dbg.gamma_applied = tiff_gamma
    dbg.tonemap = globals().get("TIFF_FLOAT_TONEMAP", "none")

    use_pillow = (tiff_reader in ("auto", "pillow"))
    use_tifffile = (tiff_reader in ("auto", "tifffile"))

    # 1) Try Pillow
    if use_pillow:
        try:
            with Image.open(path) as im:
                dbg.loader = "Pillow"
                dbg.dtype = str(getattr(im, "mode", ""))  # not super useful for float, but record it
                if tiff_apply_icc and HAVE_IMAGECMS and "icc_profile" in im.info and im.info.get("icc_profile"):
                    try:
                        src = ImageCms.ImageCmsProfile(io.BytesIO(im.info["icc_profile"]))
                        dst = ImageCms.createProfile("sRGB")
                        im = ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
                        dbg.icc_applied = True
                    except Exception:
                        im = im.convert("RGB")
                else:
                    im = im.convert("RGB")

                if (tiff_exposure_ev and tiff_exposure_ev != 0.0) or (tiff_gamma and tiff_gamma > 0):
                    arr01 = np.asarray(im, dtype=np.float32) / 255.0
                    arr01 = _apply_exposure_and_gamma_01(arr01, ev=tiff_exposure_ev, gamma=tiff_gamma)
                    im = Image.fromarray(np.clip(arr01 * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")
                setattr(im, "_qjpeg_debug", dbg.__dict__)
                return im
        except Exception:
            if tiff_reader == "pillow":
                raise
            pass

    # 2) tifffile path
    if use_tifffile:
        try:
            import tifffile as tiff
        except Exception as e:
            if tiff_reader == "tifffile":
                raise RuntimeError(f"tifffile not installed: {e}")
            tiff = None
        if tiff is not None:
            with tiff.TiffFile(str(path)) as tf:
                pages = list(tf.pages)
                if not pages:
                    raise RuntimeError("Empty TIFF (no pages).")
                page = max(pages, key=lambda p: (int(p.imagelength or 0) * int(p.imagewidth or 0)))
                # record TIFF tags for debug
                dbg.loader = "tifffile"
                try:
                    dbg.bits_per_sample = str(page.bitspersample)
                except Exception:
                    pass
                try:
                    dbg.sample_format = str(getattr(page, "sampleformat", ""))
                except Exception:
                    pass
                try:
                    pm = getattr(page, "photometric", "")
                    dbg.photometric = str(pm.name if hasattr(pm, "name") else pm)
                except Exception:
                    pass
                try:
                    arr = page.asarray(maxworkers=0)
                except Exception:
                    arr = page.asarray(out='memmap')
                    arr = np.array(arr, copy=True)

            dbg.dtype = str(arr.dtype); dbg.shape = tuple(arr.shape)
            # make globals visible to mapper and collect stats
            _old_g = globals().get("TIFF_GAMMA", None)
            _old_ev = globals().get("TIFF_EXPOSURE_EV", 0.0)
            globals()["TIFF_GAMMA"] = tiff_gamma
            globals()["TIFF_EXPOSURE_EV"] = tiff_exposure_ev
            arr8 = _to_uint8_rgb(arr, dbg=dbg)
            globals()["TIFF_GAMMA"] = _old_g
            globals()["TIFF_EXPOSURE_EV"] = _old_ev

            im = Image.fromarray(arr8, mode="RGB")
            setattr(im, "_qjpeg_debug", dbg.__dict__)
            return im

    raise RuntimeError("Failed to open TIFF with selected reader(s)")

def _open_dng_with_rawpy_or_fallback(path: Path, demosaic_name: Optional[str]) -> tuple[Image.Image, dict]:
    """
    Try rawpy with chosen demosaic; on errors (GPL3/unsupported/linear DNG), fallback to robust TIFF reader.
    """
    if not HAVE_RAWPY:
        # No rawpy: treat DNG as TIFF container
        img = _open_tiff_robust(
            path,
            tiff_apply_icc=False,
            tiff_gamma=globals().get("TIFF_GAMMA", None),
            tiff_exposure_ev=globals().get("TIFF_EXPOSURE_EV", 0.0),
            tiff_reader=globals().get("TIFF_READER", "auto"),
        )
        return img, {}

    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            name = (demosaic_name or "AHD").upper()
            # map string -> rawpy enum safely
            algo = {
                "AHD": rawpy.DemosaicAlgorithm.AHD,
                "LINEAR": rawpy.DemosaicAlgorithm.LINEAR,
                "AMAZE": getattr(rawpy.DemosaicAlgorithm, "AMAZE", rawpy.DemosaicAlgorithm.AHD),
            }.get(name, rawpy.DemosaicAlgorithm.AHD)

            try:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    output_color=rawpy.ColorSpace.sRGB,
                    gamma=(2.222, 4.5),
                    demosaic_algorithm=algo,
                    dcb_enhance=False,
                )
            except Exception as e:
                msg = str(e).lower()
                if "gpl3" in msg or "not supported" in msg:
                    # fall back to AHD if AMAZE is unavailable
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=True,
                        output_bps=8,
                        output_color=rawpy.ColorSpace.sRGB,
                        gamma=(2.222, 4.5),
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                        dcb_enhance=False,
                    )
                else:
                    raise
    except Exception:
        # linear/unsupported DNG -> treat as TIFF container
        img = _open_tiff_robust(
            path,
            tiff_apply_icc=False,
            tiff_gamma=globals().get("TIFF_GAMMA", None),
            tiff_exposure_ev=globals().get("TIFF_EXPOSURE_EV", 0.0),
            tiff_reader=globals().get("TIFF_READER", "auto"),
        )
        return img, {}

    img = Image.fromarray(rgb, mode="RGB")
    return img, {}

def load_image_as_rgb(path: Path, demosaic_name: Optional[str] = None,
                      tiff_apply_icc: bool = False,
                      tiff_gamma: Optional[float] = None,
                      tiff_exposure_ev: float = 0.0,
                      tiff_reader: str = "auto") -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """
    Load image or RAW robustly.
    Return (PIL.Image RGB, np.array RGB, info dict for metadata passthrough).
    """
    ext = path.suffix.lower()

    # RAW (DNG & friends) -> try rawpy, fallback to TIFF reader
    if ext in RAW_EXTS:
        img, info = _open_dng_with_rawpy_or_fallback(path, demosaic_name)
        arr = np.array(img)
        return img, arr, info

    # TIFFs can be tricky: use robust opener
    if ext in {".tif", ".tiff"}:
        img = _open_tiff_robust(path, tiff_apply_icc=tiff_apply_icc,
                                tiff_gamma=tiff_gamma, tiff_exposure_ev=tiff_exposure_ev,
                                tiff_reader=tiff_reader)
        info = {"exif": getattr(img, "info", {}).get("exif"),
                "icc_profile": getattr(img, "info", {}).get("icc_profile"),
                "debug": getattr(img, "_qjpeg_debug", None)}
        arr = np.array(img)
        return img, arr, info

    # Everything else via Pillow
    with Image.open(path) as img0:
        info = {
            "exif": img0.info.get("exif"),
            "icc_profile": img0.info.get("icc_profile"),
        }
        img = img0.convert("RGB")
    arr = np.array(img)
    return img, arr, info

# ----------------------------
# Metrics / Search
# ----------------------------
def ssim_threshold_search(
        src_img: Image.Image,
        src_arr: np.ndarray,
        threshold: float = 0.99,
        qmin: int = 1,
        qmax: int = 100,
        progressive: bool = False,
        subsampling: Optional[int] = None,
) -> Tuple[int, float]:
    """Binary search the lowest JPEG quality with SSIM >= threshold. Faster via downsample/luma."""
    # Prepare reference (possibly downsampled / luma)
    ds = int(globals().get("SSIM_DOWNSAMPLE", 1))
    luma_only = bool(globals().get("SSIM_LUMA_ONLY", False))
    ref = _downsample_arr(src_arr, ds)
    if luma_only:
        ref = _rgb_to_luma8(ref)
        ref_range = 255
        ref_kwargs = {}
    else:
        ref_range = 255
        ref_kwargs = dict(channel_axis=2)

    lo, hi = qmin, qmax
    best_q, best_ssim = qmax, 1.0
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        save_kwargs = dict(format="JPEG", quality=mid, optimize=bool(globals().get("SEARCH_OPTIMIZE", False)))
        if progressive: save_kwargs["progressive"] = True
        if subsampling is not None: save_kwargs["subsampling"] = subsampling
        src_img.save(buf, **save_kwargs)
        buf.seek(0)
        comp = Image.open(buf).convert("RGB")
        comp_arr = np.array(comp)
        comp_arr = _downsample_arr(comp_arr, ds)
        if luma_only:
            comp_arr = _rgb_to_luma8(comp_arr)
        val = ssim(ref, comp_arr, data_range=ref_range, **ref_kwargs)
        if val >= threshold:
            best_q, best_ssim = mid, val
            hi = mid - 1
        else:
            lo = mid + 1
    return best_q, best_ssim

def brisque_score_cv2(arr: np.ndarray, model_path: Optional[str], range_path: Optional[str]) -> Optional[float]:
    """No-reference BRISQUE via OpenCV (lower is better). Returns float or None."""
    if not (HAVE_CV2 and HAVE_CV2_BRISQUE):
        return None
    if not model_path or not range_path:
        return None
    if not (os.path.exists(model_path) and os.path.exists(range_path)):
        return None
    try:
        if arr.dtype != np.uint8:
            arr8 = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr8 = arr
        bgr = arr8[:, :, ::-1]  # RGB -> BGR
        br = cv2.quality.QualityBRISQUE_create(model_path, range_path)
        score = br.compute(bgr)[0]
        return float(score)
    except Exception:
        return None

# ----------------------------
# Metadata
# ----------------------------
def save_final_jpeg(
        dst_path: Path,
        src_img: Image.Image,
        quality: int,
        src_info: Dict[str, Any],
        progressive: bool = False,
        subsampling: Optional[int] = None,
):
    """Save JPEG at chosen quality, attaching EXIF/ICC if present."""
    ensure_dir(dst_path)
    save_kwargs = dict(format="JPEG", quality=quality, optimize=True)
    if progressive:
        save_kwargs["progressive"] = True
    if subsampling is not None:
        save_kwargs["subsampling"] = subsampling
    if src_info.get("exif"):
        save_kwargs["exif"] = src_info["exif"]
    if src_info.get("icc_profile"):
        save_kwargs["icc_profile"] = src_info["icc_profile"]
    src_img.save(dst_path, **save_kwargs)

def copy_all_metadata_with_exiftool(src_path: Path, dst_path: Path):
    """
    Copy ALL tags (EXIF/IPTC/XMP) from src to dst using exiftool, if available.
    Also sets filesystem modification/creation dates to match EXIF DateTimeOriginal.
    """
    if not EXIFTOOL_OK:
        return
    cmd = [
        "exiftool",
        "-overwrite_original",
        "-All:All",
        "-TagsFromFile", str(src_path),
        # Set filesystem dates to match EXIF DateTimeOriginal
        "-FileModifyDate<DateTimeOriginal",
        "-FileCreateDate<DateTimeOriginal",  # Works on macOS/Windows, ignored on Linux ext4
        str(dst_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

# ----------------------------
# Destination naming helpers
# ----------------------------
def short_hash(text: str, n=4) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def dest_path_for(
        src_path: Path,
        input_root: Path,
        out_root: Path,
        mirror_structure: bool,
        flat_dedupe: bool,
) -> Tuple[Optional[Path], bool]:
    """Compute destination path (.jpg) and a flag whether it collides in flat mode."""
    ext = src_path.suffix.lower()
    rel = src_path.relative_to(input_root) if mirror_structure else Path(src_path.name)
    dst_rel = rel.with_suffix(".jpg") if (ext in RAW_EXTS or ext in IMG_EXTS) else None
    if dst_rel is None:
        return None, False
    if mirror_structure:
        return out_root / dst_rel, False
    # flat mode: dedupe by adding a short hash from the relative path (without extension)
    base = dst_rel.stem
    hashed = f"{base}__{short_hash(str(rel.with_suffix('')))}.jpg" if flat_dedupe else f"{base}.jpg"
    return out_root / hashed, (not flat_dedupe)

# ----------------------------
# Per-file processing (worker-safe)
# ----------------------------
def process_one(
        src_path: str,
        input_root: str,
        out_root: str,
        ssim_thr: float,
        qmin: int,
        qmax: int,
        progressive: bool,
        subsampling: Optional[int],
        brisque_model: Optional[str],
        brisque_range: Optional[str],
        mirror_structure: bool,
        flat_dedupe: bool,
        resume: bool,
        demosaic_name: Optional[str],
        tiff_apply_icc: bool,
        tiff_gamma: Optional[float],
        tiff_exposure_ev: float,
        tiff_reader: str,
        exiftool_mode: str,
        debug: bool,
        debug_json: bool,
) -> Optional[Dict[str, Any]]:
    src_path_p = Path(src_path)
    input_root_p = Path(input_root)
    out_root_p = Path(out_root)

    dst_path, _ = dest_path_for(src_path_p, input_root_p, out_root_p, mirror_structure, flat_dedupe)
    if dst_path is None:
        return None  # not an image

    # Resume: skip if output exists and is newer and non-empty
    if resume and dst_path.exists():
        try:
            if dst_path.stat().st_size > 0 and dst_path.stat().st_mtime >= src_path_p.stat().st_mtime:
                return {"skipped": True, "src": str(src_path_p), "dst": str(dst_path)}
        except Exception:
            pass

    # Copy sidecars regardless
    copy_sidecars(src_path_p, dst_path.with_suffix(""))

    # Load & encode
    try:
        img, arr, info = load_image_as_rgb(
            src_path_p,
            demosaic_name=demosaic_name,
            tiff_apply_icc=tiff_apply_icc,
            tiff_gamma=tiff_gamma,
            tiff_exposure_ev=tiff_exposure_ev,
            tiff_reader=tiff_reader,
        )
    except Exception as e:
        return {"error": f"{src_path_p}: {e}"}

    q, ssim_val = ssim_threshold_search(
        img, arr, threshold=ssim_thr, qmin=qmin, qmax=qmax,
        progressive=progressive, subsampling=subsampling
    )
    save_final_jpeg(dst_path, img, q, info, progressive=progressive, subsampling=subsampling)
    if exiftool_mode == "all":
        copy_all_metadata_with_exiftool(src_path_p, dst_path)

    # Optional BRISQUE report
    bq = None
    if brisque_model and brisque_range and HAVE_CV2_BRISQUE:
        try:
            comp_arr = np.array(Image.open(dst_path).convert("RGB"))
            bq = brisque_score_cv2(comp_arr, brisque_model, brisque_range)
        except Exception:
            bq = None

    # Stats
    try:
        src_size = src_path_p.stat().st_size
    except Exception:
        src_size = 0
    try:
        dst_size = dst_path.stat().st_size
    except Exception:
        dst_size = 0
    saved = (1 - (dst_size / src_size)) * 100 if src_size > 0 else 0.0

    if info.get("debug") and (debug or debug_json):
        dbg = info["debug"]
        if debug_json:
            print(json.dumps({"file": str(src_path_p), **dbg}, ensure_ascii=False))
        else:
            print("[DEBUG]", src_path_p)
            print("        loader=", dbg.get("loader"),
                  "| dtype=", dbg.get("dtype"), "| shape=", dbg.get("shape"))
            print("        bits/sample=", dbg.get("bits_per_sample"),
                  "| sample_format=", dbg.get("sample_format"),
                  "| photometric=", dbg.get("photometric"))
            print(f"        src_min/max={dbg.get('src_min'):.6g}/{dbg.get('src_max'):.6g} "
                  f"| pcts({dbg.get('p_lo')},{dbg.get('p_hi')})="
                  f"{dbg.get('src_lo_val')} → {dbg.get('src_hi_val')}")
            if 'auto_ev_gain' in dbg:
                print(f"        autoEV: pct={dbg.get('auto_ev_pctval'):.6g} → gain={dbg.get('auto_ev_gain'):.3f}")
            print(f"        EV={dbg.get('ev_applied')} | gamma={dbg.get('gamma_applied')} "
                  f"| tonemap={dbg.get('tonemap')} | ICC-applied={dbg.get('icc_applied')}")
            if 'linY_p50_pre' in dbg:
                print(f"        linY p50 pre/post={dbg['linY_p50_pre']:.4f}/{dbg['linY_p50_post']:.4f}")
            print(f"        out_min/max={dbg.get('out_min'):.6g}/{dbg.get('out_max'):.6g}")

    return {
        "src": str(src_path_p),
        "dst": str(dst_path),
        "quality": q,
        "ssim": float(ssim_val),
        "brisque": bq,
        "saved_pct": saved
    }

# ----------------------------
# Folder walk + scheduling
# ----------------------------
def collect_sources(input_root: Path, allow_exts: Optional[Set[str]]) -> List[Path]:
    files: List[Path] = []
    for p in input_root.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if allow_exts is not None and ext not in allow_exts:
            continue
        if ext in RAW_EXTS or ext in IMG_EXTS:
            files.append(p)
    return files

def find_flat_collisions(files: List[Path]) -> List[Tuple[Path, Path]]:
    seen: Dict[str, Path] = {}
    dups: List[Tuple[Path, Path]] = []
    for p in files:
        name = p.name
        if name in seen:
            dups.append((seen[name], p))
        else:
            seen[name] = p
    return dups

def process_tree(
        input_root: Path,
        ssim_thr: float,
        qmin: int,
        qmax: int,
        progressive: bool,
        subsampling: Optional[int],
        brisque_model: Optional[str],
        brisque_range: Optional[str],
        mirror_structure: bool,
        flat_dedupe: bool,
        resume: bool,
        workers: int,
        no_brisque: bool,
        allow_exts: Optional[Set[str]],
        demosaic_name: Optional[str],
        show_progress: bool,
        tiff_apply_icc: bool,
        tiff_gamma: Optional[float],
        tiff_exposure_ev: float,
        tiff_reader: str,
        exiftool_mode: str,
        debug: bool,
        debug_json: bool,
):
    out_root = input_root.parent / f"{input_root.name}_compressed"
    out_root.mkdir(parents=True, exist_ok=True)

    files = collect_sources(input_root, allow_exts)

    # Duplicate names warning in flat mode
    if not mirror_structure:
        dups = find_flat_collisions(files)
        if dups:
            print(f"[INFO] Detected {len(dups)} duplicate basenames in flat mode.")
            print("       They will be auto-deduped with short hashes (e.g., name__abcd.jpg).")

    if no_brisque:
        brisque_model = None
        brisque_range = None

    # Schedule
    results: List[Optional[Dict[str, Any]]] = []
    total = len(files)
    print(f"Discovered {total} image(s). Processing with {workers} worker(s)...")

    # Progress bar / ETA setup
    start = time.time()
    use_bar = (show_progress and HAVE_TQDM and total > 0)
    pbar = tqdm(total=total, unit="img") if use_bar else None

    def _tick():
        if use_bar:
            elapsed = time.time() - start
            sofar = pbar.n + 1  # +1 for this tick
            rate = sofar / elapsed if elapsed > 0 else 0.0
            eta = (total - sofar) / rate if rate > 0 else 0.0
            pbar.update(1)
            pbar.set_postfix_str(
                f"elapsed {int(elapsed//60)}m{int(elapsed%60):02d}s | eta {int(eta//60)}m{int(eta%60):02d}s"
            )
        else:
            # quiet fallback; per-file lines still print
            pass

    task_kwargs = dict(
        input_root=str(input_root),
        out_root=str(out_root),
        ssim_thr=ssim_thr,
        qmin=qmin,
        qmax=qmax,
        progressive=progressive,
        subsampling=subsampling,
        brisque_model=brisque_model,
        brisque_range=brisque_range,
        mirror_structure=mirror_structure,
        flat_dedupe=flat_dedupe,
        resume=resume,
        demosaic_name=demosaic_name,
        tiff_apply_icc=tiff_apply_icc,
        tiff_gamma=tiff_gamma,
        tiff_exposure_ev=tiff_exposure_ev,
        tiff_reader=tiff_reader,
        exiftool_mode=exiftool_mode,
        debug=debug,
        debug_json=debug_json,
    )

    if workers <= 1:
        for f in files:
            res = process_one(str(f), **task_kwargs)
            if res and "error" in res:
                print(f"[ERR] {res['error']}")
            elif res and res.get("skipped"):
                print(f"[SKIP] {res['src']}")
            elif res:
                print(f"[OK] {res['src']} -> {res['dst']}  "
                      f"quality={res['quality']}, SSIM={res['ssim']:.4f}"
                      + (f", BRISQUE={res['brisque']:.2f}" if res['brisque'] is not None else "")
                      + (f" | saved {res['saved_pct']:.1f}%" if 'saved_pct' in res and res['saved_pct'] is not None else ""))
            _tick()
            results.append(res)
    else:
        # Collect global settings to pass to worker processes
        import functools
        settings_dict = {
            "TIFF_SMART16": globals().get("TIFF_SMART16", False),
            "TIFF_SMART16_PCTS": globals().get("TIFF_SMART16_PCTS", (0.5, 99.5)),
            "TIFF_SMART16_PERCHANNEL": globals().get("TIFF_SMART16_PERCHANNEL", False),
            "SMART16_DOWNSAMPLE": globals().get("SMART16_DOWNSAMPLE", 1),
            "SSIM_DOWNSAMPLE": globals().get("SSIM_DOWNSAMPLE", 1),
            "SSIM_LUMA_ONLY": globals().get("SSIM_LUMA_ONLY", False),
            "SEARCH_OPTIMIZE": globals().get("SEARCH_OPTIMIZE", False),
            "TIFF_GAMMA": globals().get("TIFF_GAMMA", None),
            "TIFF_EXPOSURE_EV": globals().get("TIFF_EXPOSURE_EV", 0.0),
            "TIFF_FLOAT_TONEMAP": globals().get("TIFF_FLOAT_TONEMAP", "none"),
            "AUTO_EV_MODE": globals().get("AUTO_EV_MODE", "off"),
            "AUTO_EV_MID": globals().get("AUTO_EV_MID", 0.18),
            "AUTO_EV_MID_PCT": globals().get("AUTO_EV_MID_PCT", 50.0),
            "AUTO_EV_HI_PCT": globals().get("AUTO_EV_HI_PCT", 99.0),
            "AUTO_EV_HI_CAP": globals().get("AUTO_EV_HI_CAP", 0.90),
            "AUTO_EV_DOWNSAMPLE": globals().get("AUTO_EV_DOWNSAMPLE", 8),
            "AUTO_EV_BOUNDS": globals().get("AUTO_EV_BOUNDS", (-4.0, 6.0)),
            "AUTO_EV_ITERS": globals().get("AUTO_EV_ITERS", 16),
            "BLACKPOINT_PCT": globals().get("BLACKPOINT_PCT", None),
            "WHITEPOINT_PCT": globals().get("WHITEPOINT_PCT", None),
            "SHADOW_LIFT": globals().get("SHADOW_LIFT", 0.0),
            "CONTRAST_STRENGTH": globals().get("CONTRAST_STRENGTH", 0.0),
            "SATURATION": globals().get("SATURATION", 1.0),
        }
        initializer = functools.partial(_init_worker_globals, settings_dict)

        with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as ex:
            futs = [ex.submit(process_one, str(f), **task_kwargs) for f in files]
            for fut in as_completed(futs):
                res = fut.result()
                if res and "error" in res:
                    print(f"[ERR] {res['error']}")
                elif res and res.get("skipped"):
                    print(f"[SKIP] {res['src']}")
                elif res:
                    print(f"[OK] {res['src']} -> {res['dst']}  "
                          f"quality={res['quality']}, SSIM={res['ssim']:.4f}"
                          + (f", BRISQUE={res['brisque']:.2f}" if res['brisque'] is not None else "")
                          + (f" | saved {res['saved_pct']:.1f}%" if 'saved_pct' in res and res['saved_pct'] is not None else ""))
                _tick()
                results.append(res)

    # Close progress bar
    if 'pbar' in locals() and pbar is not None:
        pbar.close()

    # Summary
    done = sum(1 for r in results if r and not r.get("skipped") and "error" not in r)
    skipped = sum(1 for r in results if r and r.get("skipped"))
    failed = sum(1 for r in results if r and "error" in r)
    print("\n=== Summary ===")
    print(f"Input root:  {input_root}")
    print(f"Output root: {out_root}")
    print(f"Processed:   {done}  |  Skipped: {skipped}  |  Failed: {failed}")

    return results

# ----------------------------
# Utility: play a sound when done
# ----------------------------

def _play_finish_sound():
    try:
        snd = Path(__file__).resolve().with_name("microwave-ding-104123.mp3")
        if not snd.exists():
            return
        # Prefer platform native players to avoid adding dependencies
        player = None
        args = None
        # macOS
        if shutil.which("afplay"):
            player = "afplay"
            args = [player, str(snd)]
        # Linux options
        elif shutil.which("paplay"):
            player = "paplay"
            args = [player, str(snd)]
        elif shutil.which("aplay"):
            player = "aplay"
            args = [player, str(snd)]
        elif shutil.which("ffplay"):
            player = "ffplay"
            args = [player, "-nodisp", "-autoexit", str(snd)]
        # Fallback to playsound if available (may not support all platforms reliably)
        if args is not None:
            try:
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        try:
            from playsound import playsound  # type: ignore
            import threading
            threading.Thread(target=lambda: playsound(str(snd)), daemon=True).start()
        except Exception:
            # Last resort: do nothing
            pass
    except Exception:
        # Never let the notifier crash the main script
        pass

# ----------------------------
# Config file support
# ----------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and return dict of settings."""
    if not HAVE_YAML:
        print("[WARNING] PyYAML not installed. Config files require: pip install pyyaml")
        return {}

    p = Path(config_path)
    if not p.exists():
        # Try looking in presets/ directory
        preset_path = SCRIPT_DIR / "presets" / config_path
        if not preset_path.exists():
            preset_path = SCRIPT_DIR / "presets" / f"{config_path}.yaml"
        if preset_path.exists():
            p = preset_path
        else:
            print(f"[WARNING] Config file not found: {config_path}")
            return {}

    try:
        with open(p, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[CONFIG] Loaded: {p}")
        return config if config else {}
    except Exception as e:
        print(f"[WARNING] Failed to load config {p}: {e}")
        return {}

def save_config(config_path: str, args: argparse.Namespace):
    """Save current args to a YAML config file."""
    if not HAVE_YAML:
        print("[ERROR] PyYAML not installed. Cannot save config.")
        return

    # Convert args to dict, excluding None values and input_root
    config = {}
    for key, value in vars(args).items():
        if key == 'input_root' or key == 'config' or key == 'save_config':
            continue
        if value is not None and value != argparse.SUPPRESS:
            # Skip default values for cleaner config
            if key in ['qmin', 'qmax'] and value in [1, 100]:
                continue

            # Convert comma-separated strings back to lists for proper YAML formatting
            if key == 'tiff_smart16_pct' and isinstance(value, str) and ',' in value:
                parts = [float(x.strip()) for x in value.split(',')]
                config[key] = parts
            elif key == 'auto_ev_bounds' and isinstance(value, str) and ',' in value:
                parts = [float(x.strip()) for x in value.split(',')]
                config[key] = parts
            else:
                config[key] = value

    try:
        p = Path(config_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            # Use custom representer for lists to make them inline [x, y] instead of block style
            class FlowListDumper(yaml.SafeDumper):
                pass
            def represent_list(dumper, data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            FlowListDumper.add_representer(list, represent_list)

            yaml.dump(config, f, Dumper=FlowListDumper, default_flow_style=False, sort_keys=False)
        print(f"[CONFIG] Saved to: {p}")
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=("Batch JPEG optimizer with SSIM target; robust RAW/TIFF loading; "
                     "metadata/sidecars preserved; OpenCV BRISQUE optional; multiprocessing; "
                     "resume; de-dup for flat mode; file-type filtering; smart 16-bit scaling.")
    )
    p.add_argument("input_root", type=str, help="Folder to process recursively.")
    p.add_argument("--ssim", type=float, default=0.99, help="SSIM threshold to maintain (default: 0.99).")
    p.add_argument("--qmin", type=int, default=1, help="Minimum JPEG quality to consider (default: 1).")
    p.add_argument("--qmax", type=int, default=100, help="Maximum JPEG quality to consider (default: 100).")
    p.add_argument("--flat", action="store_true", help="Do NOT mirror subfolder structure (outputs in root_compressed).")
    p.add_argument("--progressive", action="store_true", help="Save progressive JPEGs.")
    p.add_argument("--subsampling", type=int, choices=[0, 1, 2], default=None,
                   help="Force chroma subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0. Default: Pillow decides.")
    # SSIM/search speedups
    p.add_argument("--ssim-downsample", type=int, default=4,
                   help="Compute SSIM on a 1/N grid (e.g., 4 → img[::4,::4]). 1 disables downsampling.")
    p.add_argument("--ssim-luma-only", action="store_true",
                   help="Compute SSIM on luma only (Y), faster and usually sufficient.")
    p.add_argument("--search-optimize", action="store_true",
                   help="Use Pillow optimize=True during quality search (slower). Default: off.")
    p.add_argument("--brisque-model", type=str, default=os.environ.get("BRISQUE_MODEL", DEFAULT_BRISQUE_MODEL),
                   help="Path to BRISQUE_model_live.yml")
    p.add_argument("--brisque-range", type=str, default=os.environ.get("BRISQUE_RANGE", DEFAULT_BRISQUE_RANGE),
                   help="Path to BRISQUE_range_live.yml")
    p.add_argument("--no-brisque", action="store_true", help="Disable BRISQUE scoring to speed up.")
    p.add_argument("--exiftool-mode", type=str, choices=["all","none"], default="all",
                   help="Copy metadata with exiftool after save. 'none' skips the external call (faster).")
    p.add_argument("--resume", action="store_true", help="Skip files whose outputs already exist and are up-to-date.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2),
                   help="Parallel workers (default: half your CPUs).")
    p.add_argument("--types", type=str, default="",
                   help="Comma-separated list of extensions to process (e.g. 'tif,tiff,dng,jpg'). If empty, process all supported.")
    p.add_argument("--tiff-smart16", action="store_true",
                   help="Enable smart 16-bit TIFF scaling (percentile stretch to 8-bit).")
    p.add_argument("--tiff-smart16-pct", type=str, default="0.5,99.5",
                   help="Low,High percentiles for smart 16-bit scaling (default: '0.5,99.5').")
    p.add_argument("--tiff-smart16-perchannel", action="store_true",
                   help="With --tiff-smart16, stretch each RGB channel independently (more punch, may shift color slightly). If omitted, uses a single global curve for all channels (safer colors).")
    p.add_argument("--smart16-downsample", type=int, default=8,
                   help="Subsample step for percentile stretch (e.g., 8 → use every 8th pixel).")
    p.add_argument("--demosaic", type=str, default="AHD",
                   choices=["AHD", "LINEAR", "AMAZE"],
                   help="RAW demosaic algorithm (default: AHD). AMAZE needs GPL3 libraw; will fallback if unavailable.")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable progress bar / ETA output.")
    # New TIFF handling options
    p.add_argument("--tiff-apply-icc", action="store_true",
                   help="If TIFF has an embedded ICC profile, convert to sRGB using it (Pillow path only).")
    p.add_argument("--tiff-gamma", type=float, default=None,
                   help="Apply display gamma to linear TIFFs after 16→8 normalization (e.g., 2.2).")
    p.add_argument("--tiff-exposure-ev", type=float, default=0.0,
                   help="Exposure compensation in EV (applied before gamma; e.g., 1.0 doubles brightness).")
    p.add_argument("--tiff-reader", type=str, choices=["auto", "pillow", "tifffile"], default="auto",
                   help=("Which TIFF loader to use. 'auto' tries Pillow then falls back to tifffile. "
                         "'tifffile' forces percentile smart16 path; 'pillow' uses ICC and EV/Gamma only."))
    # Debug and tonemapping options
    p.add_argument("--debug", action="store_true", help="Print per-file mapping details.")
    p.add_argument("--debug-json", action="store_true", help="Emit per-file debug as JSON lines.")
    p.add_argument("--tiff-float-tonemap", type=str, choices=["none", "reinhard", "aces"],
                   default="none", help="Optional tone mapping for float TIFFs (applied after EV, before gamma).")
    # New Auto-EV options
    p.add_argument("--auto-ev-mode", type=str, choices=["off","mid","mid_guard"], default="off",
                   help="Auto exposure per image. 'mid' matches a mid-tone target; 'mid_guard' also caps highlights.")
    p.add_argument("--auto-ev-mid", type=float, default=0.18,
                   help="Target mid luminance (after tonemap, before gamma). Typical 0.16–0.22.")
    p.add_argument("--auto-ev-mid-pct", type=float, default=50.0,
                   help="Which luminance percentile to anchor for the mid (default median=50).")
    p.add_argument("--auto-ev-hi-pct", type=float, default=99.0,
                   help="Highlight percentile to protect (default 99.0).")
    p.add_argument("--auto-ev-hi-cap", type=float, default=0.90,
                   help="Max allowed highlight luminance after tonemap (default 0.90).")
    p.add_argument("--auto-ev-downsample", type=int, default=8,
                   help="Subsample step for EV solving (e.g., 8 means img[::8,::8]).")
    p.add_argument("--auto-ev-bounds", type=str, default="-4,6",
                   help="EV search bounds as 'lo,hi' (default -4..+6).")
    p.add_argument("--auto-ev-iters", type=int, default=16,
                   help="Bisection iterations (default 16).")
    # Back-compat (no longer used):
    p.add_argument("--auto-ev-percentile", type=float, default=50.0,
                   help="[DEPRECATED] Old auto-EV percentile option (ignored if --auto-ev-mode is used).")
    # Post-gamma shaping options
    p.add_argument("--blackpoint-pct", type=float, default=None,
                   help="After gamma, map this luminance percentile to 0 (e.g., 0.2).")
    p.add_argument("--whitepoint-pct", type=float, default=None,
                   help="After gamma, map this luminance percentile to 1 (e.g., 99.7).")
    p.add_argument("--shadows", type=float, default=0.0,
                   help="Shadow lift amount (0=no change, 0.1–0.3 brightens dark areas without affecting highlights).")
    p.add_argument("--contrast", type=float, default=0.0,
                   help="Post-gamma S-curve strength (0=no change, try 0.08–0.20).")
    p.add_argument("--saturation", type=float, default=1.0,
                   help="Post-gamma saturation multiplier (1=no change, e.g., 1.05).")
    # Config file support
    p.add_argument("--config", type=str, default=None,
                   help="Load settings from YAML config file. Can be a path or preset name (e.g., 'hdr-default').")
    p.add_argument("--save-config", type=str, default=None,
                   help="Save current settings to YAML config file and exit.")

    # Two-pass parsing: load config first, then override with CLI args
    # First pass: parse to check for --config
    args_temp, _ = p.parse_known_args()

    # Load config if specified
    config_dict = {}
    if args_temp.config:
        config_dict = load_config(args_temp.config)

    # Apply config as defaults
    if config_dict:
        # Convert config values to argparse defaults
        for key, value in config_dict.items():
            # Handle special cases
            if key == 'tiff_smart16_pct' and isinstance(value, list):
                config_dict[key] = f"{value[0]},{value[1]}"
            elif key == 'auto_ev_bounds' and isinstance(value, list):
                config_dict[key] = f"{value[0]},{value[1]}"

        # Set defaults from config
        for action in p._actions:
            if action.dest in config_dict and action.dest != 'config':
                # Convert underscores to match config keys
                config_key = action.dest.replace('_', '_')
                if config_key in config_dict:
                    action.default = config_dict[config_key]

    # Second pass: parse all args (CLI args override config)
    args = p.parse_args()

    # Handle save-config
    if args.save_config:
        save_config(args.save_config, args)
        import sys
        sys.exit(0)

    return args

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root not found or not a directory: {input_root}")

    # --- Normalize file-type filter ---
    if args.types.strip():
        _exts = [e.strip().lower() for e in args.types.split(",") if e.strip()]
        ALLOW_EXTS: Optional[Set[str]] = {e if e.startswith(".") else f".{e}" for e in _exts}
    else:
        ALLOW_EXTS = None  # allow all supported

    # --- TIFF smart 16-bit scaling from CLI ---
    TIFF_SMART16 = bool(args.tiff_smart16)
    try:
        lo_s, hi_s = (args.tiff_smart16_pct.split(",") if isinstance(args.tiff_smart16_pct, str)
                      else ("0.5", "99.5"))
        TIFF_SMART16_PCTS = (float(lo_s), float(hi_s))
    except Exception:
        TIFF_SMART16_PCTS = (0.5, 99.5)

    # --- TIFF smart16 per-channel toggle ---
    TIFF_SMART16_PERCHANNEL = bool(getattr(args, "tiff_smart16_perchannel", False))
    SMART16_DOWNSAMPLE = max(1, int(getattr(args, "smart16_downsample", 8)))

    # Search/SSIM globals
    SSIM_DOWNSAMPLE = max(1, int(getattr(args, "ssim_downsample", 1)))
    SSIM_LUMA_ONLY = bool(getattr(args, "ssim_luma_only", False))
    SEARCH_OPTIMIZE = bool(getattr(args, "search_optimize", False))

    # make these visible to _to_uint8_rgb
    TIFF_GAMMA = args.tiff_gamma  # None or float (e.g., 2.2)
    TIFF_EXPOSURE_EV = float(getattr(args, "tiff_exposure_ev", 0.0) or 0.0)
    TIFF_READER = args.tiff_reader
    TIFF_FLOAT_TONEMAP = args.tiff_float_tonemap
    DEBUG_ON = bool(args.debug or args.debug_json)
    DEBUG_JSON = bool(args.debug_json)

    # --- Auto-EV globals ---
    AUTO_EV_MODE = args.auto_ev_mode
    AUTO_EV_MID = args.auto_ev_mid
    AUTO_EV_MID_PCT = args.auto_ev_mid_pct
    AUTO_EV_HI_PCT = args.auto_ev_hi_pct
    AUTO_EV_HI_CAP = args.auto_ev_hi_cap
    AUTO_EV_DOWNSAMPLE = max(1, int(args.auto_ev_downsample))
    try:
        _lo, _hi = (args.auto_ev_bounds.split(",") if isinstance(args.auto_ev_bounds, str) else ("-4","6"))
        AUTO_EV_BOUNDS = (float(_lo), float(_hi))
    except Exception:
        AUTO_EV_BOUNDS = (-4.0, 6.0)
    AUTO_EV_ITERS = int(args.auto_ev_iters)

    # Post-gamma shaping globals
    BLACKPOINT_PCT = args.blackpoint_pct
    WHITEPOINT_PCT = args.whitepoint_pct
    # Safety net: if user didn't pass either, enable sensible defaults to avoid the gray veil
    if BLACKPOINT_PCT is None and WHITEPOINT_PCT is None:
        BLACKPOINT_PCT = 0.6
        WHITEPOINT_PCT = 99.7
        print("[INFO] Enabled default post-gamma anchoring: blackpoint 0.6 / whitepoint 99.7 to prevent gray veil.\n       Override with --blackpoint-pct/--whitepoint-pct or set either to disable.")
    SHADOW_LIFT = args.shadows
    CONTRAST_STRENGTH = args.contrast
    SATURATION = args.saturation

    # --- Diagnostics ---
    print(f"exiftool detected: {'YES' if EXIFTOOL_OK else 'NO'}")
    if HAVE_CV2:
        print(f"OpenCV: {cv2.__version__}")
        print(f"cv2.quality present: {'YES' if HAVE_CV2_QUALITY else 'NO'}")
        print(f"OpenCV BRISQUE available: {'YES' if HAVE_CV2_BRISQUE else 'NO'}")
    else:
        print("OpenCV not importable at all.")

    # Show which model files will be used
    mp = args.brisque_model
    rp = args.brisque_range
    print(f"BRISQUE model file: {mp} ({'FOUND' if os.path.exists(mp) else 'MISSING'})")
    print(f"BRISQUE range file: {rp} ({'FOUND' if os.path.exists(rp) else 'MISSING'})\n")

    try:
        process_tree(
            input_root=input_root,
            ssim_thr=args.ssim,
            qmin=args.qmin,
            qmax=args.qmax,
            progressive=args.progressive,
            subsampling=args.subsampling,
            brisque_model=args.brisque_model,
            brisque_range=args.brisque_range,
            mirror_structure=(not args.flat),
            flat_dedupe=args.flat,     # only used when flat mode is on
            resume=args.resume,
            workers=max(1, args.workers),
            no_brisque=args.no_brisque,
            allow_exts=ALLOW_EXTS,
            demosaic_name=args.demosaic,
            show_progress=(not args.no_progress),
            tiff_apply_icc=args.tiff_apply_icc,
            tiff_gamma=args.tiff_gamma,
            tiff_exposure_ev=args.tiff_exposure_ev,
            tiff_reader=args.tiff_reader,
            exiftool_mode=args.exiftool_mode,
            debug=args.debug,
            debug_json=args.debug_json,
        )
    finally:
        _play_finish_sound()
