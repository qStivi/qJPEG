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

try:
    from scipy import ndimage
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

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


def detect_moire_fft(arr: np.ndarray, threshold: float = 0.10, min_peak_ratio: float = 1.5) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Detect moiré patterns using FFT frequency analysis.

    Moiré patterns create characteristic peaks in the frequency domain due to
    aliasing between the camera sensor grid and display pixel grid. This function
    detects such patterns by analyzing the 2D FFT magnitude spectrum.

    Args:
        arr: uint8 RGB array (will be converted to grayscale internally)
        threshold: Confidence threshold for detection (0-1)
                  Higher values = less sensitive (fewer false positives)
                  Default 0.10 balances sensitivity and specificity
        min_peak_ratio: Minimum ratio of peak magnitude to median magnitude
                       Screen moiré typically creates peaks 1.5-3x above background
                       Lower values = more sensitive but more false positives

    Returns:
        Tuple of (has_moire, confidence, debug_info):
        - has_moire: Boolean indicating if moiré pattern detected
        - confidence: Float 0-1 indicating detection confidence
        - debug_info: Dict with diagnostic information

    Examples:
        >>> # Detect moiré in a screen photograph
        >>> has_moire, conf, info = detect_moire_fft(screen_photo_arr)
        >>> if has_moire:
        ...     print(f"Moiré detected with {conf:.1%} confidence")
        ...     print(f"Peak count: {info['peak_count']}")
    """
    debug = {
        'available': HAVE_SCIPY and cv2 is not None,
        'peak_count': 0,
        'max_peak_strength': 0.0,
        'median_magnitude': 0.0,
        'detection_method': 'fft_frequency_analysis'
    }

    # Early exit if scipy unavailable
    if not HAVE_SCIPY:
        return False, 0.0, debug

    try:
        # Convert to grayscale for frequency analysis
        if arr.ndim == 3:
            # Use luminance weights for proper grayscale conversion
            gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        else:
            gray = arr.astype(np.float32)

        debug['shape'] = gray.shape

        # Apply 2D FFT and compute magnitude spectrum
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)  # Center DC component
        magnitude = np.abs(fft_shifted)

        # Log scale for better visualization and analysis
        magnitude_log = np.log1p(magnitude)

        # Mask out DC component and immediate neighbors (always strong)
        h, w = magnitude_log.shape
        cy, cx = h // 2, w // 2
        dc_mask_radius = max(h, w) // 50  # Mask ~2% of spectrum
        y, x = np.ogrid[:h, :w]
        dc_mask = ((y - cy)**2 + (x - cx)**2 <= dc_mask_radius**2)
        magnitude_masked = magnitude_log.copy()
        magnitude_masked[dc_mask] = 0

        # Compute statistics
        median_mag = np.median(magnitude_masked[magnitude_masked > 0])
        debug['median_magnitude'] = float(median_mag)

        # Normalize magnitude spectrum
        if median_mag > 0:
            magnitude_norm = magnitude_masked / median_mag
        else:
            return False, 0.0, debug

        # Detect peaks using threshold
        # Moiré patterns create strong, isolated peaks away from DC
        # Peaks should be at least min_peak_ratio times the median
        peaks = magnitude_norm > min_peak_ratio
        peak_count = np.sum(peaks)
        debug['peak_count'] = int(peak_count)

        if peak_count > 0:
            max_peak = np.max(magnitude_norm[peaks])
            debug['max_peak_strength'] = float(max_peak)

            # Calculate confidence based on peak strength and count
            # More peaks + stronger peaks = higher confidence
            strength_factor = min(max_peak / 10.0, 1.0)  # Normalize to 0-1
            count_factor = min(peak_count / 20.0, 1.0)   # 20+ peaks = very likely
            confidence = (strength_factor + count_factor) / 2.0

            has_moire = confidence >= threshold
            debug['confidence_raw'] = float(confidence)

            return has_moire, confidence, debug

        return False, 0.0, debug

    except Exception as e:
        debug['error'] = str(e)
        return False, 0.0, debug


def brisque_score_with_moire_check(
    arr: np.ndarray,
    model_path: Optional[str],
    range_path: Optional[str],
    check_moire: bool = True,
    moire_threshold: float = 0.10
) -> Tuple[Optional[float], bool, Dict[str, Any]]:
    """
    Compute BRISQUE score with optional moiré pattern detection.

    This wrapper extends brisque_score_cv2() to detect and flag images
    with moiré patterns, which can produce unreliable BRISQUE scores.

    Args:
        arr: uint8 RGB array
        model_path: Path to BRISQUE model YAML
        range_path: Path to BRISQUE range YAML
        check_moire: Whether to run moiré detection (default True)
        moire_threshold: Threshold for moiré detection (0-1)

    Returns:
        Tuple of (brisque_score, unreliable_flag, debug_info):
        - brisque_score: BRISQUE score (float) or None if unavailable
        - unreliable_flag: True if moiré detected (score may be unreliable)
        - debug_info: Dict with moiré detection details

    Examples:
        >>> score, unreliable, info = brisque_score_with_moire_check(
        ...     img_arr, "model.yml", "range.yml"
        ... )
        >>> if unreliable:
        ...     print(f"Warning: BRISQUE={score:.1f} may be unreliable (moiré detected)")
    """
    # Compute BRISQUE score
    brisque = brisque_score_cv2(arr, model_path, range_path)

    # Check for moiré patterns if requested
    unreliable = False
    moire_info = {}

    if check_moire:
        has_moire, confidence, debug = detect_moire_fft(arr, threshold=moire_threshold)
        unreliable = has_moire
        moire_info = {
            'moire_detected': has_moire,
            'moire_confidence': confidence,
            'moire_debug': debug
        }

    return brisque, unreliable, moire_info


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
