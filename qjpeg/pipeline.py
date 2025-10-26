"""
Processing pipeline orchestration for qJPEG.

Coordinates the processing of entire directory trees with multiprocessing support.
"""

import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set, List, Dict, Any
import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

try:
    import cv2
    HAVE_CV2_BRISQUE = hasattr(cv2, "quality") and hasattr(cv2.quality, "QualityBRISQUE_create")
except Exception:
    HAVE_CV2_BRISQUE = False

from .utils import dest_path_for, collect_sources, find_flat_collisions
from .metadata import copy_sidecars, copy_all_metadata_with_exiftool
from .image_io import load_image_as_rgb
from .quality import ssim_threshold_search, brisque_score_cv2, save_final_jpeg


def _init_worker_globals(settings_dict):
    """
    Initialize global variables in worker processes.

    When using multiprocessing, each worker process starts fresh and doesn't
    inherit the parent's global state. This function sets up all necessary
    globals in each worker.

    Args:
        settings_dict: Dict of global variable names and values
    """
    global TIFF_SMART16, TIFF_SMART16_PCTS, TIFF_SMART16_PERCHANNEL
    global SMART16_DOWNSAMPLE, SSIM_DOWNSAMPLE, SSIM_LUMA_ONLY, SEARCH_OPTIMIZE
    global TIFF_GAMMA, TIFF_EXPOSURE_EV, TIFF_FLOAT_TONEMAP
    global AUTO_EV_MODE, AUTO_EV_MID, AUTO_EV_MID_PCT, AUTO_EV_HI_PCT, AUTO_EV_HI_CAP
    global AUTO_EV_DOWNSAMPLE, AUTO_EV_BOUNDS, AUTO_EV_ITERS
    global BLACKPOINT_PCT, WHITEPOINT_PCT, SHADOW_LIFT, CONTRAST_STRENGTH, SATURATION

    for key, value in settings_dict.items():
        globals()[key] = value


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
    """
    Process a single image file.

    Args:
        src_path: Source image path (as string for multiprocessing)
        input_root: Input root directory
        out_root: Output root directory
        ssim_thr: SSIM threshold to maintain
        qmin: Minimum JPEG quality
        qmax: Maximum JPEG quality
        progressive: Use progressive JPEG
        subsampling: Chroma subsampling
        brisque_model: Path to BRISQUE model
        brisque_range: Path to BRISQUE range
        mirror_structure: Mirror directory structure
        flat_dedupe: Deduplicate flat filenames
        resume: Skip existing outputs
        demosaic_name: RAW demosaic algorithm
        tiff_apply_icc: Apply ICC conversion for TIFFs
        tiff_gamma: TIFF display gamma
        tiff_exposure_ev: TIFF exposure compensation
        tiff_reader: TIFF reader ('auto', 'pillow', 'tifffile')
        exiftool_mode: Metadata copy mode ('all', 'none')
        debug: Print debug info
        debug_json: Print debug as JSON

    Returns:
        Result dict with quality, SSIM, size savings, etc.
    """
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
    """
    Process entire directory tree of images.

    Creates output directory, collects source files, and processes them
    with optional multiprocessing.

    Args:
        (see process_one for most parameters)
        input_root: Root directory to process
        workers: Number of parallel workers
        no_brisque: Disable BRISQUE scoring
        allow_exts: Set of allowed file extensions, or None for all
        show_progress: Show progress bar

    Returns:
        List of result dicts
    """
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


def _play_finish_sound():
    """Play a sound notification when processing completes."""
    try:
        snd = Path(__file__).resolve().parent.parent / "microwave-ding-104123.mp3"
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
        # Fallback to playsound if available
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
