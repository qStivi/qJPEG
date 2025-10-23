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

import numpy as np
from PIL import Image, ImageFile
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
def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert various TIFF array dtypes/shapes to 8-bit RGB (optionally with smart 16-bit percentile stretch)."""
    if arr.dtype == np.uint16:
        if TIFF_SMART16:
            lo, hi = TIFF_SMART16_PCTS
            arrf = arr.astype(np.float32)
            if arrf.ndim == 2:  # gray
                lo_v, hi_v = np.percentile(arrf, (lo, hi))
                if hi_v <= lo_v:
                    arr8 = (arr >> 8).astype(np.uint8)
                else:
                    arrf = np.clip((arrf - lo_v) / (hi_v - lo_v), 0.0, 1.0)
                    arr8 = (arrf * 255.0 + 0.5).astype(np.uint8)
            else:
                # color image
                h, w = arrf.shape[:2]
                c = arrf.shape[2] if arrf.ndim == 3 else 1
                if c == 1:
                    arr8 = (arr >> 8).astype(np.uint8)
                else:
                    out = np.empty((h, w, 3), np.uint8)
                    if TIFF_SMART16_PERCHANNEL:
                        # per-channel stretch: can increase pop/saturation, may shift color slightly
                        for ch in range(min(c, 3)):
                            lo_v, hi_v = np.percentile(arrf[..., ch], (lo, hi))
                            if hi_v <= lo_v:
                                out[..., ch] = (arr[..., ch] >> 8).astype(np.uint8)
                            else:
                                chf = np.clip((arrf[..., ch] - lo_v) / (hi_v - lo_v), 0.0, 1.0)
                                out[..., ch] = (chf * 255.0 + 0.5).astype(np.uint8)
                    else:
                        # global stretch: same curve for all channels, preserves color ratios better
                        lo_v, hi_v = np.percentile(arrf, (lo, hi))
                        if hi_v <= lo_v:
                            out = (arr[..., :3] >> 8).astype(np.uint8)
                        else:
                            arrn = np.clip((arrf - lo_v) / (hi_v - lo_v), 0.0, 1.0)
                            out = (arrn[..., :3] * 255.0 + 0.5).astype(np.uint8)

                    if c < 3:
                        for ch in range(c, 3):
                            out[..., ch] = out[..., c-1]
                    arr8 = out
        else:
            arr8 = (arr >> 8).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        arr8 = np.clip(arr, 0.0, 1.0)
        arr8 = (arr8 * 255.0 + 0.5).astype(np.uint8)
    else:
        arr8 = arr.astype(np.uint8, copy=False)

    # Normalize channels to 3
    if arr8.ndim == 2:  # gray
        return np.stack([arr8, arr8, arr8], axis=-1)
    if arr8.ndim == 3:
        h, w, c = arr8.shape
        if c == 1:
            return np.repeat(arr8, 3, axis=-1)
        if c >= 3:
            return arr8[:, :, :3]
    return np.repeat(arr8[..., None], 3, axis=-1)

def _open_tiff_robust(path: Path) -> Image.Image:
    """Try Pillow first; fall back to tifffile with safer reading for truncated/odd TIFFs."""
    # 1) Try Pillow
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        pass

    # 2) tifffile fallback
    try:
        import tifffile as tiff
    except Exception as e:
        raise RuntimeError(f"Cannot open TIFF with Pillow and tifffile not installed: {e}")

    # attempt simple imread first (single array)
    try:
        arr = tiff.imread(str(path), maxworkers=0)
        arr8 = _to_uint8_rgb(arr)
        return Image.fromarray(arr8, mode="RGB")
    except Exception:
        pass

    # manual pages walk as last resort
    with tiff.TiffFile(str(path)) as tf:
        pages = list(tf.pages)
        if not pages:
            raise RuntimeError("Empty TIFF (no pages).")
        # choose largest page by pixel count
        page = max(pages, key=lambda p: (int(p.imagelength or 0) * int(p.imagewidth or 0)))
        try:
            arr = page.asarray(maxworkers=0)
        except Exception:
            # some very broken files: try memmap then copy
            arr = page.asarray(out='memmap')
            arr = np.array(arr, copy=True)
    arr8 = _to_uint8_rgb(arr)
    return Image.fromarray(arr8, mode="RGB")

def _open_dng_with_rawpy_or_fallback(path: Path, demosaic_name: Optional[str]) -> tuple[Image.Image, dict]:
    """
    Try rawpy with chosen demosaic; on errors (GPL3/unsupported/linear DNG), fallback to robust TIFF reader.
    """
    if not HAVE_RAWPY:
        # No rawpy: treat DNG as TIFF container
        img = _open_tiff_robust(path)
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
        img = _open_tiff_robust(path)
        return img, {}

    img = Image.fromarray(rgb, mode="RGB")
    return img, {}

def load_image_as_rgb(path: Path, demosaic_name: Optional[str] = None) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
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
        img = _open_tiff_robust(path)
        info = {"exif": getattr(img, "info", {}).get("exif"),
                "icc_profile": getattr(img, "info", {}).get("icc_profile")}
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
        img, arr, info = load_image_as_rgb(src_path_p, demosaic_name=demosaic_name)
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
    )
