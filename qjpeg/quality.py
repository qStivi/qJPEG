"""
Quality assessment and JPEG encoding for qJPEG.

SSIM-guided quality search and optional BRISQUE scoring.
"""

import io
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

try:
    import cv2
    HAVE_CV2 = True
    HAVE_CV2_QUALITY = hasattr(cv2, "quality")
    HAVE_CV2_BRISQUE = HAVE_CV2_QUALITY and hasattr(cv2.quality, "QualityBRISQUE_create")
except Exception:
    HAVE_CV2 = False
    HAVE_CV2_QUALITY = False
    HAVE_CV2_BRISQUE = False

from .utils import ensure_dir
from .image_processing import _downsample_arr, _rgb_to_luma8


def ssim_threshold_search(
        src_img: Image.Image,
        src_arr: np.ndarray,
        threshold: float = 0.99,
        qmin: int = 1,
        qmax: int = 100,
        progressive: bool = False,
        subsampling: Optional[int] = None,
) -> Tuple[int, float]:
    """
    Binary search the lowest JPEG quality with SSIM >= threshold.

    Uses downsampling and optional luma-only comparison for speed.

    Args:
        src_img: Source PIL Image
        src_arr: Source array (uint8 RGB)
        threshold: Minimum SSIM to maintain (e.g., 0.99)
        qmin: Minimum quality to test
        qmax: Maximum quality to test
        progressive: Use progressive JPEG
        subsampling: Chroma subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0)

    Returns:
        (best_quality, achieved_ssim)
    """
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
    """
    Compute no-reference BRISQUE quality score via OpenCV.

    Lower scores indicate better perceptual quality.

    Args:
        arr: uint8 RGB array
        model_path: Path to BRISQUE model YAML
        range_path: Path to BRISQUE range YAML

    Returns:
        BRISQUE score (float), or None if unavailable
    """
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


def save_final_jpeg(
        dst_path: Path,
        src_img: Image.Image,
        quality: int,
        src_info: Dict[str, Any],
        progressive: bool = False,
        subsampling: Optional[int] = None,
):
    """
    Save JPEG at chosen quality, attaching EXIF/ICC if present.

    Args:
        dst_path: Destination path for JPEG
        src_img: Source PIL Image
        quality: JPEG quality (1-100)
        src_info: Metadata dict with 'exif' and 'icc_profile' keys
        progressive: Use progressive JPEG
        subsampling: Chroma subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0)
    """
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
