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
TIFF_SMART16_PERCHANNEL: bool = True  # default to per-channel stretch; set False to use global curve
# Exposure/Gamma globals for 16-bit TIFF mapping (set in __main__ from CLI)
TIFF_GAMMA: Optional[float] = None
TIFF_EXPOSURE_EV: float = 0.0


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
    if arrf.ndim == 2:  # grayscale
        lo_v, hi_v = np.nanpercentile(arrf, (lo, hi))
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
            lo_v, hi_v = np.nanpercentile(arrf[..., ch], (lo, hi))
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
        lo_v, hi_v = np.nanpercentile(arrf[..., :c], (lo, hi))
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

def _to_uint8_rgb(arr: np.ndarray, dbg: Optional[MapStats] = None) -> np.ndarray:
    """
    Convert TIFF array (uint16 or float) to 8-bit RGB, honoring:
    - smart percentile stretch (also for float!)
    - EV and gamma
    - optional tonemap for float
    Fills dbg with min/max and lo/hi stats when provided.
    """
    ev = globals().get("TIFF_EXPOSURE_EV", 0.0)
    gamma = globals().get("TIFF_GAMMA", None)
    tonemap_mode = globals().get("TIFF_FLOAT_TONEMAP", "none")
    use_smart = bool(globals().get("TIFF_SMART16", False))
    per_channel = bool(globals().get("TIFF_SMART16_PERCHANNEL", True))
    lo, hi = globals().get("TIFF_SMART16_PCTS", (0.5, 99.5))

    # ---------------- uint16 ----------------
    if arr.dtype == np.uint16:
        arrf = arr.astype(np.float32)
        if use_smart:
            arr01, lo_v, hi_v = _percentile_stretch(arrf, lo, hi, per_channel)
        else:
            arr01 = arrf / 65535.0
            if arr01.ndim == 2:
                arr01 = arr01[..., None]
            lo_v = 0.0; hi_v = 65535.0
        # EV + gamma (display)
        if ev:
            arr01 = arr01 * (2.0 ** ev)
        arr01 = np.clip(arr01, 0.0, 1.0)
        if gamma and gamma > 0:
            arr01 = np.power(arr01, 1.0 / gamma)

        if dbg:
            dbg.src_min = float(arrf.min()); dbg.src_max = float(arrf.max())
            dbg.p_lo = lo; dbg.p_hi = hi; dbg.per_channel = per_channel
            dbg.src_lo_val = lo_v; dbg.src_hi_val = hi_v
            dbg.ev_applied = ev; dbg.gamma_applied = gamma
            dbg.tonemap = "none"
            dbg.out_min = float(arr01.min()); dbg.out_max = float(arr01.max())

        if arr01.shape[-1] == 1:
            arr01 = np.repeat(arr01, 3, axis=-1)
        return (arr01 * 255.0 + 0.5).astype(np.uint8)

    # ---------------- float32/64 ----------------
    if np.issubdtype(arr.dtype, np.floating):
        arrf = arr.astype(np.float32)
        # Robust range (like smart16), now applied to float as well
        if use_smart:
            arr01, lo_v, hi_v = _percentile_stretch(arrf, lo, hi, per_channel)
        else:
            # if user disables smart, try to normalize by the visible energy
            # assume typical linear floats are around [0,1] but may exceed
            lo_v, hi_v = float(np.nanmin(arrf)), float(np.nanpercentile(arrf, 99.9))
            rng = max(1e-6, hi_v - lo_v)
            arr01 = np.clip((arrf - lo_v) / rng, 0.0, 1.0)
            if arr01.ndim == 2:
                arr01 = arr01[..., None]

        # Exposure first (still linear)
        if ev:
            arr_lin = arr01 * (2.0 ** ev)
        else:
            arr_lin = arr01

        # Optional tonemap while still linear and before gamma
        arr_lin = np.maximum(arr_lin, 0.0)
        arr_tm = _tonemap(arr_lin, tonemap_mode)

        # Clip (after tonemap) and apply display gamma if requested
        arr_tm = np.clip(arr_tm, 0.0, 1.0)
        if gamma and gamma > 0:
            arr_disp = np.power(arr_tm, 1.0 / gamma)
        else:
            arr_disp = arr_tm

        if dbg:
            dbg.src_min = float(np.nanmin(arrf)); dbg.src_max = float(np.nanmax(arrf))
            dbg.p_lo = lo; dbg.p_hi = hi; dbg.per_channel = per_channel
            dbg.src_lo_val = lo_v; dbg.src_hi_val = hi_v
            dbg.ev_applied = ev; dbg.gamma_applied = gamma; dbg.tonemap = tonemap_mode
            dbg.out_min = float(arr_disp.min()); dbg.out_max = float(arr_disp.max())

        if arr_disp.shape[-1] == 1:
            arr_disp = np.repeat(arr_disp, 3, axis=-1)
        return (arr_disp * 255.0 + 0.5).astype(np.uint8)

    # ---------------- fallback: already 8/16 int ----------------
    arr8 = arr.astype(np.uint8, copy=False)
    if arr8.ndim == 2:
        return np.stack([arr8, arr8, arr8], axis=-1)
    if arr8.ndim == 3 and arr8.shape[-1] >= 3:
        return arr8[:, :, :3]
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
    """Binary search the lowest JPEG quality with SSIM >= threshold. Returns (quality, ssim_val)."""
    lo, hi = qmin, qmax
    best_q = qmax
    best_ssim = 1.0
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        save_kwargs = dict(format="JPEG", quality=mid, optimize=True)
        if progressive:
            save_kwargs["progressive"] = True
        if subsampling is not None:
            # subsampling=0 -> 4:4:4, 1 -> 4:2:2, 2 -> 4:2:0 (Pillow)
            save_kwargs["subsampling"] = subsampling
        src_img.save(buf, **save_kwargs)
        buf.seek(0)
        comp = Image.open(buf)
        comp_arr = np.array(comp)
        val = ssim(src_arr, comp_arr, data_range=255, channel_axis=2)
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
    """Copy ALL tags (EXIF/IPTC/XMP) from src to dst using exiftool, if available."""
    if not EXIFTOOL_OK:
        return
    cmd = [
        "exiftool",
        "-overwrite_original",
        "-All:All",
        "-TagsFromFile", str(src_path),
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

    if tiff_reader and info.get("debug") and (tiff_reader in ("tifffile", "auto")):
        dbg = info["debug"]
        if debug:
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
                print(f"        EV={dbg.get('ev_applied')} | gamma={dbg.get('gamma_applied')} "
                      f"| tonemap={dbg.get('tonemap')} | ICC-applied={dbg.get('icc_applied')}")
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
        with ProcessPoolExecutor(max_workers=workers) as ex:
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
    p.add_argument("--brisque-model", type=str, default=os.environ.get("BRISQUE_MODEL", DEFAULT_BRISQUE_MODEL),
                   help="Path to BRISQUE_model_live.yml")
    p.add_argument("--brisque-range", type=str, default=os.environ.get("BRISQUE_RANGE", DEFAULT_BRISQUE_RANGE),
                   help="Path to BRISQUE_range_live.yml")
    p.add_argument("--no-brisque", action="store_true", help="Disable BRISQUE scoring to speed up.")
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
    return p.parse_args()

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
    TIFF_SMART16_PERCHANNEL = bool(getattr(args, "tiff_smart16_perchannel", False)) or TIFF_SMART16_PERCHANNEL

    # make these visible to _to_uint8_rgb
    TIFF_GAMMA = args.tiff_gamma  # None or float (e.g., 2.2)
    TIFF_EXPOSURE_EV = float(getattr(args, "tiff_exposure_ev", 0.0) or 0.0)
    TIFF_READER = args.tiff_reader
    TIFF_FLOAT_TONEMAP = args.tiff_float_tonemap
    DEBUG_ON = bool(args.debug or args.debug_json)
    DEBUG_JSON = bool(args.debug_json)

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
        debug=args.debug,
        debug_json=args.debug_json,
    )
