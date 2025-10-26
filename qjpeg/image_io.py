"""
Image loading for qJPEG.

Robust loading of TIFF, RAW (DNG/CR2/NEF/etc), and standard images.
Handles 8-bit, 16-bit, 32-bit float TIFFs with proper tone mapping.
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

try:
    from PIL import ImageCms
    HAVE_IMAGECMS = True
except Exception:
    HAVE_IMAGECMS = False

try:
    import rawpy
    HAVE_RAWPY = True
except Exception:
    HAVE_RAWPY = False

from .utils import RAW_EXTS
from .image_processing import MapStats, _to_uint8_rgb, _apply_exposure_and_gamma_01


def _open_tiff_robust(path: Path, tiff_apply_icc: bool, tiff_gamma: Optional[float], tiff_exposure_ev: float,
                      tiff_reader: str = "auto") -> Image.Image:
    """
    Robustly open TIFF files (8-bit, 16-bit, or 32-bit float).

    Tries Pillow first (with ICC support), then falls back to tifffile for
    complex TIFFs (float32, high-bit-depth).

    Args:
        path: Path to TIFF file
        tiff_apply_icc: Convert embedded ICC profile to sRGB
        tiff_gamma: Display gamma (e.g., 2.2)
        tiff_exposure_ev: Exposure compensation in EV
        tiff_reader: 'auto', 'pillow', or 'tifffile'

    Returns:
        PIL Image in RGB mode
    """
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
                dbg.dtype = str(getattr(im, "mode", ""))
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

    Args:
        path: Path to DNG/RAW file
        demosaic_name: 'AHD', 'LINEAR', or 'AMAZE'

    Returns:
        (PIL Image, metadata dict)
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

    Handles:
    - RAW files (DNG, CR2, CR3, NEF, ARW, etc.) via rawpy
    - TIFF files (8/16/32-bit) via Pillow or tifffile
    - Standard images (JPEG, PNG, etc.) via Pillow

    Args:
        path: Path to image file
        demosaic_name: RAW demosaic algorithm ('AHD', 'LINEAR', 'AMAZE')
        tiff_apply_icc: Apply ICC profile conversion for TIFFs
        tiff_gamma: Display gamma for TIFFs
        tiff_exposure_ev: Exposure compensation for TIFFs
        tiff_reader: TIFF reader ('auto', 'pillow', 'tifffile')

    Returns:
        (PIL.Image RGB, np.array RGB, metadata dict)
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
