#!/usr/bin/env python3
"""
Analyze TIFF pipeline to debug gray veil issue.
Usage: python analyze_tiff.py path/to/image.TIF
"""
import sys
import numpy as np
from pathlib import Path

try:
    import tifffile
except ImportError:
    print("ERROR: pip install tifffile")
    sys.exit(1)

def analyze_tiff(path):
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(path).name}")
    print(f"{'='*60}\n")

    # Load raw data
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        print("TIFF Metadata:")
        print(f"  Bits/sample: {page.bitspersample}")
        print(f"  Sample format: {page.sampleformat} (1=uint, 2=int, 3=float)")
        print(f"  Photometric: {page.photometric}")
        print(f"  Shape: {page.shape}")
        print(f"  Dtype: {page.dtype}")

        arr = page.asarray()

    print(f"\nRaw Data Range:")
    print(f"  Min: {np.nanmin(arr):.6f}")
    print(f"  Max: {np.nanmax(arr):.6f}")
    print(f"  Mean: {np.nanmean(arr):.6f}")
    print(f"  Median: {np.nanmedian(arr):.6f}")

    # Percentiles
    print(f"\nRaw Percentiles (per-channel):")
    for ch, name in enumerate(['R', 'G', 'B']):
        if arr.ndim == 3 and arr.shape[-1] > ch:
            p = np.nanpercentile(arr[..., ch], [0.5, 1, 2, 5, 50, 95, 98, 99, 99.5])
            print(f"  {name}: 0.5%={p[0]:.4f} 1%={p[1]:.4f} 2%={p[2]:.4f} 5%={p[3]:.4f} | "
                  f"median={p[4]:.4f} | 95%={p[5]:.4f} 98%={p[6]:.4f} 99%={p[7]:.4f} 99.5%={p[8]:.4f}")

    # Global percentiles
    p_global = np.nanpercentile(arr, [0.5, 1, 2, 5, 50, 95, 98, 99, 99.5])
    print(f"  Global: 0.5%={p_global[0]:.4f} 1%={p_global[1]:.4f} 2%={p_global[2]:.4f} "
          f"5%={p_global[3]:.4f} | median={p_global[4]:.4f} | "
          f"95%={p_global[5]:.4f} 98%={p_global[6]:.4f} 99%={p_global[7]:.4f} 99.5%={p_global[8]:.4f}")

    # Simulate percentile stretch (current code)
    print(f"\n{'='*60}")
    print("Simulating Percentile Stretch (0.5%, 99.0%)")
    print(f"{'='*60}")

    arrf = arr.astype(np.float32)

    # Per-channel stretch
    print(f"\nPer-Channel Stretch:")
    arr_perchan = np.empty_like(arrf)
    for ch, name in enumerate(['R', 'G', 'B']):
        if arr.ndim == 3 and arr.shape[-1] > ch:
            lo_v, hi_v = np.nanpercentile(arrf[..., ch], (0.5, 99.0))
            print(f"  {name}: lo={lo_v:.6f}, hi={hi_v:.6f}, range={hi_v-lo_v:.6f}")
            if hi_v > lo_v:
                arr_perchan[..., ch] = np.clip((arrf[..., ch] - lo_v) / (hi_v - lo_v), 0.0, 1.0)
            else:
                arr_perchan[..., ch] = np.clip(arrf[..., ch], 0.0, 1.0)

    print(f"  After stretch - min: {arr_perchan.min():.6f}, max: {arr_perchan.max():.6f}")

    # ACES tonemap simulation
    print(f"\n{'='*60}")
    print("After Auto-EV + ACES Tonemap + Gamma 2.2")
    print(f"{'='*60}")

    # Apply typical auto-EV (~1.14 from your debug)
    ev = 1.14
    arr_ev = arr_perchan * (2.0 ** ev)
    print(f"\nAfter EV +{ev}:")
    print(f"  Min: {arr_ev.min():.6f}, Max: {arr_ev.max():.6f}")

    # ACES tonemap
    def aces(arr):
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((arr*(a*arr + b)) / (arr*(c*arr + d) + e), 0.0, 1.0)

    arr_tm = aces(np.clip(arr_ev, 0, None))
    print(f"\nAfter ACES tonemap:")
    print(f"  Min: {arr_tm.min():.6f}, Max: {arr_tm.max():.6f}")

    # Gamma
    arr_gamma = np.power(arr_tm, 1.0 / 2.2)
    print(f"\nAfter Gamma 2.2:")
    print(f"  Min: {arr_gamma.min():.6f}, Max: {arr_gamma.max():.6f}")

    # Luma for black/white point
    Y = 0.2126*arr_gamma[...,0] + 0.7152*arr_gamma[...,1] + 0.0722*arr_gamma[...,2]

    print(f"\n{'='*60}")
    print("Post-Gamma Luminance Percentiles (for black/white point)")
    print(f"{'='*60}")

    luma_p = np.nanpercentile(Y, [0, 0.2, 1, 2, 3, 4, 5, 50, 95, 96, 97, 98, 99, 99.8, 100])
    print(f"  0%: {luma_p[0]:.6f}")
    print(f"  0.2%: {luma_p[1]:.6f} ← Your original blackpoint")
    print(f"  1%: {luma_p[2]:.6f}")
    print(f"  2%: {luma_p[3]:.6f} ← Updated blackpoint")
    print(f"  3%: {luma_p[4]:.6f}")
    print(f"  4%: {luma_p[5]:.6f}")
    print(f"  5%: {luma_p[6]:.6f}")
    print(f"  50%: {luma_p[7]:.6f} (median)")
    print(f"  95%: {luma_p[8]:.6f}")
    print(f"  96%: {luma_p[9]:.6f}")
    print(f"  97%: {luma_p[10]:.6f} ← Updated whitepoint")
    print(f"  98%: {luma_p[11]:.6f} ← Updated whitepoint")
    print(f"  99%: {luma_p[12]:.6f}")
    print(f"  99.8%: {luma_p[13]:.6f} ← Your original whitepoint")
    print(f"  100%: {luma_p[14]:.6f}")

    # Simulate black/white point remap
    print(f"\n{'='*60}")
    print("Testing Black/White Point Remapping")
    print(f"{'='*60}")

    for bp_pct, wp_pct in [(0.2, 99.8), (2.0, 98.0), (3.0, 97.0), (4.0, 96.0), (5.0, 95.0)]:
        bp_val = np.nanpercentile(Y, bp_pct)
        wp_val = np.nanpercentile(Y, wp_pct)

        if wp_val > bp_val:
            arr_remap = np.clip((arr_gamma - bp_val) / (wp_val - bp_val), 0.0, 1.0)
            print(f"\n  {bp_pct}% / {wp_pct}%: bp_val={bp_val:.6f}, wp_val={wp_val:.6f}")
            print(f"    After remap: min={arr_remap.min():.6f}, max={arr_remap.max():.6f}")

            # Check if it actually reaches 0 and 1
            if arr_remap.min() < 0.001:
                print(f"    ✓ Blacks good (min < 0.001)")
            else:
                print(f"    ✗ Blacks lifted to {arr_remap.min():.6f}")

            if arr_remap.max() > 0.999:
                print(f"    ✓ Whites good (max > 0.999)")
            else:
                print(f"    ✗ Whites dimmed to {arr_remap.max():.6f}")

    # Simulate shadow lift effect
    print(f"\n{'='*60}")
    print("Shadow Lift Effect (after black/white remap)")
    print(f"{'='*60}")

    bp_val = np.nanpercentile(Y, 3.0)
    wp_val = np.nanpercentile(Y, 97.0)
    arr_remap = np.clip((arr_gamma - bp_val) / (wp_val - bp_val), 0.0, 1.0)

    for shadow_lift in [0.0, 0.10, 0.15, 0.20, 0.25]:
        arr_test = arr_remap.copy()
        if shadow_lift > 0:
            Y_remap = 0.2126*arr_test[...,0] + 0.7152*arr_test[...,1] + 0.0722*arr_test[...,2]
            mask = np.power(np.clip(1.0 - Y_remap, 0.0, 1.0), 2.0)
            arr_test = np.clip(arr_test + shadow_lift * mask[..., None], 0.0, 1.0)

        print(f"\n  Shadow lift {shadow_lift:.2f}:")
        print(f"    Min: {arr_test.min():.6f}, Max: {arr_test.max():.6f}")
        if arr_test.min() > 0.01:
            print(f"    ✗ Blacks lifted by shadow lift!")

    # Simulate contrast S-curve effect
    print(f"\n{'='*60}")
    print("Contrast S-Curve Effect (after black/white remap)")
    print(f"{'='*60}")

    for contrast in [0.0, 0.06, 0.08, 0.10]:
        arr_test = arr_remap.copy()
        if contrast > 0:
            gain = 2.0 + 4.0*contrast*2.0
            arr_test = 0.5 + np.tanh((arr_test - 0.5) * gain) * 0.5

        print(f"\n  Contrast {contrast:.2f}:")
        print(f"    Min: {arr_test.min():.6f}, Max: {arr_test.max():.6f}")
        if arr_test.max() < 0.98:
            print(f"    ✗ Whites compressed by S-curve!")

    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}\n")

    if luma_p[1] < 0.01:  # 0.2 percentile is very dark
        print("✓ Your images are already very dark at low percentiles")
        print("  Recommended: Use HIGHER percentiles for black/white point")
        print(f"  Try: --blackpoint-pct 3.0 --whitepoint-pct 97.0")

    if luma_p[14] < 0.97:  # Max luminance after gamma is <0.97
        print("\n✗ Your images don't reach full white even at 100th percentile!")
        print("  This suggests the issue is BEFORE the black/white point remap")
        print("  Problem might be in: percentile stretch, tonemap, or gamma")

    print("\n✗ Shadow lift and contrast S-curve RE-ADD gray veil!")
    print("  Recommendation: Set --shadows 0.0 --contrast 0.0")
    print("  Or apply them BEFORE black/white point remap, not after")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_tiff.py path/to/image.TIF")
        sys.exit(1)

    analyze_tiff(sys.argv[1])
