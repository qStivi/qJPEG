#!/usr/bin/env python3
"""
qJPEG - Intelligent JPEG optimizer with HDR/RAW support.

Batch JPEG optimizer with SSIM-guided quality search, robust RAW/TIFF loading,
metadata preservation, and optional BRISQUE scoring.

Usage examples:
  python main.py "/path/to/Camera Roll" --ssim 0.95 --workers 8 --resume
  python main.py "/path/to/Camera Roll" --config float32
  python main.py "/path/to/Camera Roll" --types tif,tiff,dng --tiff-smart16
"""

import os
import sys
from pathlib import Path
from typing import Optional, Set

# Import qJPEG package
import qjpeg
from qjpeg import parse_args, process_tree, _play_finish_sound
from qjpeg import image_processing, image_io, quality, pipeline

# Check for dependencies
try:
    import cv2
    HAVE_CV2 = True
    HAVE_CV2_QUALITY = hasattr(cv2, "quality")
    HAVE_CV2_BRISQUE = HAVE_CV2_QUALITY and hasattr(cv2.quality, "QualityBRISQUE_create")
except Exception:
    HAVE_CV2 = False
    HAVE_CV2_QUALITY = False
    HAVE_CV2_BRISQUE = False


def set_globals_from_args(args):
    """
    Set global variables in qjpeg modules based on parsed arguments.

    This function updates globals in multiple modules to make settings
    available throughout the processing pipeline.
    """
    # --- Normalize file-type filter ---
    if args.types.strip():
        _exts = [e.strip().lower() for e in args.types.split(",") if e.strip()]
        allow_exts: Optional[Set[str]] = {e if e.startswith(".") else f".{e}" for e in _exts}
    else:
        allow_exts = None  # allow all supported

    # --- TIFF smart 16-bit scaling from CLI ---
    tiff_smart16 = bool(args.tiff_smart16)
    try:
        lo_s, hi_s = (args.tiff_smart16_pct.split(",") if isinstance(args.tiff_smart16_pct, str)
                      else ("0.5", "99.5"))
        tiff_smart16_pcts = (float(lo_s), float(hi_s))
    except Exception:
        tiff_smart16_pcts = (0.5, 99.5)

    # --- TIFF smart16 per-channel toggle ---
    tiff_smart16_perchannel = bool(getattr(args, "tiff_smart16_perchannel", False))
    smart16_downsample = max(1, int(getattr(args, "smart16_downsample", 8)))

    # Search/SSIM globals
    ssim_downsample = max(1, int(getattr(args, "ssim_downsample", 1)))
    ssim_luma_only = bool(getattr(args, "ssim_luma_only", False))
    search_optimize = bool(getattr(args, "search_optimize", False))

    # TIFF processing globals
    tiff_gamma = args.tiff_gamma  # None or float (e.g., 2.2)
    tiff_exposure_ev = float(getattr(args, "tiff_exposure_ev", 0.0) or 0.0)
    tiff_reader = args.tiff_reader
    tiff_float_tonemap = args.tiff_float_tonemap

    # --- Auto-EV globals ---
    auto_ev_mode = args.auto_ev_mode
    auto_ev_mid = args.auto_ev_mid
    auto_ev_mid_pct = args.auto_ev_mid_pct
    auto_ev_hi_pct = args.auto_ev_hi_pct
    auto_ev_hi_cap = args.auto_ev_hi_cap
    auto_ev_downsample = max(1, int(args.auto_ev_downsample))
    try:
        _lo, _hi = (args.auto_ev_bounds.split(",") if isinstance(args.auto_ev_bounds, str) else ("-4","6"))
        auto_ev_bounds = (float(_lo), float(_hi))
    except Exception:
        auto_ev_bounds = (-4.0, 6.0)
    auto_ev_iters = int(args.auto_ev_iters)

    # Post-gamma shaping globals
    blackpoint_pct = args.blackpoint_pct
    whitepoint_pct = args.whitepoint_pct
    # Safety net: if user didn't pass either, enable sensible defaults to avoid the gray veil
    if blackpoint_pct is None and whitepoint_pct is None:
        blackpoint_pct = 0.6
        whitepoint_pct = 99.7
        print("[INFO] Enabled default post-gamma anchoring: blackpoint 0.6 / whitepoint 99.7 to prevent gray veil.\n       Override with --blackpoint-pct/--whitepoint-pct or set either to disable.")
    shadow_lift = args.shadows
    contrast_strength = args.contrast
    saturation = args.saturation

    # Set globals in all relevant modules
    modules = [image_processing, image_io, quality, pipeline]
    for mod in modules:
        mod.TIFF_SMART16 = tiff_smart16
        mod.TIFF_SMART16_PCTS = tiff_smart16_pcts
        mod.TIFF_SMART16_PERCHANNEL = tiff_smart16_perchannel
        mod.SMART16_DOWNSAMPLE = smart16_downsample
        mod.SSIM_DOWNSAMPLE = ssim_downsample
        mod.SSIM_LUMA_ONLY = ssim_luma_only
        mod.SEARCH_OPTIMIZE = search_optimize
        mod.TIFF_GAMMA = tiff_gamma
        mod.TIFF_EXPOSURE_EV = tiff_exposure_ev
        mod.TIFF_READER = tiff_reader
        mod.TIFF_FLOAT_TONEMAP = tiff_float_tonemap
        mod.AUTO_EV_MODE = auto_ev_mode
        mod.AUTO_EV_MID = auto_ev_mid
        mod.AUTO_EV_MID_PCT = auto_ev_mid_pct
        mod.AUTO_EV_HI_PCT = auto_ev_hi_pct
        mod.AUTO_EV_HI_CAP = auto_ev_hi_cap
        mod.AUTO_EV_DOWNSAMPLE = auto_ev_downsample
        mod.AUTO_EV_BOUNDS = auto_ev_bounds
        mod.AUTO_EV_ITERS = auto_ev_iters
        mod.BLACKPOINT_PCT = blackpoint_pct
        mod.WHITEPOINT_PCT = whitepoint_pct
        mod.SHADOW_LIFT = shadow_lift
        mod.CONTRAST_STRENGTH = contrast_strength
        mod.SATURATION = saturation

    return allow_exts


def main():
    """Main entry point for qJPEG."""
    # Parse command-line arguments
    args = parse_args()

    # Validate input directory
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root not found or not a directory: {input_root}")

    # Set up global configuration
    allow_exts = set_globals_from_args(args)

    # --- Diagnostics ---
    from qjpeg.metadata import EXIFTOOL_OK
    print(f"qJPEG version: {qjpeg.__version__}")
    print(f"exiftool detected: {'YES' if EXIFTOOL_OK else 'NO'}")
    if HAVE_CV2:
        print(f"OpenCV: {cv2.__version__}")
        print(f"cv2.quality present: {'YES' if HAVE_CV2_QUALITY else 'NO'}")
        print(f"OpenCV BRISQUE available: {'YES' if HAVE_CV2_BRISQUE else 'NO'}")
    else:
        print("OpenCV not importable.")

    # Show which model files will be used
    mp = args.brisque_model
    rp = args.brisque_range
    print(f"BRISQUE model file: {mp} ({'FOUND' if os.path.exists(mp) else 'MISSING'})")
    print(f"BRISQUE range file: {rp} ({'FOUND' if os.path.exists(rp) else 'MISSING'})\n")

    # Process directory tree
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
            allow_exts=allow_exts,
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


if __name__ == "__main__":
    main()
