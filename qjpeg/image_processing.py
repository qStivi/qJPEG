"""
Image processing and transformations for qJPEG.

HDR tone mapping, auto-exposure, percentile stretching, gamma correction,
and post-processing (contrast, saturation, black/white point).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MapStats:
    """Statistics for image processing pipeline debugging."""
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
    """
    Stretch array to [0,1] using percentile values.

    Args:
        arrf: Input float array
        lo: Low percentile (e.g., 0.5)
        hi: High percentile (e.g., 99.5)
        per_channel: If True, stretch each RGB channel independently

    Returns:
        (stretched_array, lo_values, hi_values)
    """
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
    """
    Tone map HDR linear data to [0,1] range.

    Args:
        arr: Linear non-negative array
        mode: 'none', 'reinhard', or 'aces'

    Returns:
        Tone mapped array in [0,1]
    """
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
    """
    Compute scene-linear luminance using Rec. 709 coefficients.

    Args:
        a: RGB array in [0,1] with shape (..., 3)

    Returns:
        Luminance array with shape (...)
    """
    return 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]


def _subsample(arr: np.ndarray, step: int) -> np.ndarray:
    """Subsample array by taking every Nth pixel."""
    return arr[::step, ::step, :] if arr.ndim == 3 else arr[::step, ::step]


def _downsample_arr(a: np.ndarray, step: int) -> np.ndarray:
    """Downsample 2D array."""
    return a[::step, ::step] if step > 1 else a


def _rgb_to_luma8(a: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to uint8 luma."""
    return (0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]).astype(np.uint8)


def _bisect_ev(f, target: float, lo: float, hi: float, iters: int) -> float:
    """
    Find EV where f(EV) ~= target using bisection.

    Args:
        f: Monotonic function of EV
        target: Target value
        lo: Low bound for EV search
        hi: High bound for EV search
        iters: Number of iterations

    Returns:
        EV value where f(EV) ≈ target
    """
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
    Solve auto-exposure to match mid-tone target and protect highlights.

    Args:
        arr01_lin: Percentile-stretched linear RGB in [0,1] BEFORE EV
        tonemap_mode: Tone mapping mode ('none', 'reinhard', 'aces')
        mid_target: Target mid luminance after tonemap (e.g., 0.18)
        mid_pct: Percentile for mid-tone (e.g., 50.0 for median)
        hi_pct: Percentile for highlight protection (e.g., 99.0), or None
        hi_cap: Max allowed highlight luminance (e.g., 0.90), or None
        ds: Downsample step for speed
        bounds: (lo_ev, hi_ev) search bounds
        iters: Bisection iterations

    Returns:
        Dict with 'ev_mid', 'ev_hi', 'ev_final', 'mid_val', 'hi_val'
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
    """
    Apply post-gamma shaping: black/white point, shadow lift, contrast, saturation.

    Args:
        arr_disp: Display-space RGB in [0,1] (after gamma)
        dbg: Optional MapStats for debugging

    Returns:
        Shaped array in [0,1]
    """
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
    Apply exposure (EV) and gamma correction.

    Args:
        arr01: Float32 array in [0,1]
        ev: Exposure compensation in EV stops
        gamma: Display gamma (e.g., 2.2), or None for no gamma

    Returns:
        Display-space array in [0,1]
    """
    if ev and ev != 0.0:
        arr01 = arr01 * (2.0 ** ev)
    arr01 = np.clip(arr01, 0.0, 1.0)
    if gamma and gamma > 0:
        arr01 = np.power(arr01, 1.0 / gamma)  # linear -> display
    return arr01


def _to_uint8_rgb(arr: np.ndarray, dbg: MapStats | None = None) -> np.ndarray:
    """
    Map uint16/float TIFF data to 8-bit RGB.

    Pipeline:
      - Percentile stretch (for uint16/float)
      - Optional auto-EV anchor
      - Manual EV compensation
      - Tone mapping (for HDR float data)
      - Gamma correction
      - Post-gamma shaping (black/white point, contrast, saturation)

    Args:
        arr: Input array (uint16, float32, uint8)
        dbg: Optional MapStats for debugging

    Returns:
        uint8 RGB array
    """
    ev = globals().get("TIFF_EXPOSURE_EV", 0.0)
    gamma = globals().get("TIFF_GAMMA", None)
    tonemap_mode = globals().get("TIFF_FLOAT_TONEMAP", "none")
    use_smart = bool(globals().get("TIFF_SMART16", False))
    per_channel = bool(globals().get("TIFF_SMART16_PERCHANNEL", False))
    lo, hi = globals().get("TIFF_SMART16_PCTS", (0.5, 99.5))

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
