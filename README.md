# qJPEG — Intelligent JPEG Optimizer & Converter

## Overview
**qJPEG** is a perceptually‑aware batch optimizer that (a) finds the lowest JPEG quality meeting a user target **SSIM**, and (b) safely converts high‑bit‑depth TIFF/RAW/HDR sources to good‑looking SDR JPEGs while preserving metadata. It scales to entire libraries (mirrored or flat output), is resilient to odd files, and runs fast with multiprocessing.

## NEW: Simplified Workflow (v2.0)
**No more 20+ parameter commands!** Use the new features:

1. **Config File System** - Use presets instead of long command lines:
   ```bash
   python main.py "/path/to/photos" --config hdr-default
   ```

2. **Interactive Calibration** - Automatically find optimal settings:
   ```bash
   python calibrate.py "/path/to/photos" --output my-settings.yaml
   ```

3. **Shadow Lift** - Brighten dark subjects without blowing highlights:
   ```bash
   python main.py "/path/to/photos" --config hdr-default --shadows 0.20
   ```

**[→ See QUICK_START.md for the simple guide](QUICK_START.md)**

---

## What’s new in this revision
This README reflects all work done in the latest code you shared.

**Tone & HDR mapping**
- Percentile‑based **Smart16** works for both **uint16 and float32/float64** TIFFs.
- Optional **tone mapping** for float/linear content: `--tiff-float-tonemap {none|reinhard|aces}` (applied after EV, before gamma).
- **Exposure & Gamma** controls for TIFF paths: `--tiff-exposure-ev`, `--tiff-gamma`.
- New **post‑gamma shaping** (display space):
  - `--blackpoint-pct` and `--whitepoint-pct` map luminance percentiles to 0/1 → kills "gray veil" and restores punch.
  - `--contrast` gentle S‑curve.
  - `--saturation` multiplier in display space.

**Auto‑EV (per‑image) exposure**
- `--auto-ev-mode {off|mid|mid_guard}`
  - **mid**: matches a mid‑tone luminance target (`--auto-ev-mid`, default 0.18) at a chosen percentile (`--auto-ev-mid-pct`).
  - **mid_guard**: as above, but *also* caps highlights (percentile `--auto-ev-hi-pct`) to a ceiling `--auto-ev-hi-cap` (default 0.90). Great for skies/clouds.
- Tunables for speed/robustness: `--auto-ev-downsample`, `--auto-ev-bounds`, `--auto-ev-iters`.
- Works in conjunction with a **manual global offset** (`--tiff-exposure-ev`) if you want a set‑wide bias.

**Speedups & throughput**
- SSIM search can be **downsampled** and/or computed on **luma only**:
  - `--ssim-downsample N` (e.g., 4 → use every 4th pixel in each dimension)
  - `--ssim-luma-only`
- Smart16 percentile estimation can be **subsampled**: `--smart16-downsample N`.
- Multiprocessing via `--workers`. Progress bar with ETA (tqdm). `--resume` to skip done files.
- Optional BRISQUE disabled with `--no-brisque` for max speed.

**I/O, color & metadata**
- Robust TIFF loader path: Pillow → tifffile fallback, with embedded **ICC → sRGB** (if `--tiff-apply-icc`).
- RAW/DNG via **rawpy** (camera WB, sRGB, no auto‑bright, AHD/LINEAR/AMAZE where available). Fallback to TIFF path for linear/unsupported DNGs.
- Sidecars mirrored (.xmp/.xml/.json). EXIF/ICC carried by Pillow; *optionally* copy **all** tags via exiftool:
  - `--exiftool-mode {all|none}` (default **all**).

**Quality/search ergonomics**
- Standard JPEG options: `--progressive`, `--subsampling {0|1|2}`.
- Search toggle: `--search-optimize` uses Pillow optimize during quality search (slower; default off).

---

## The processing pipeline (mental model)
1. **Load** (Pillow or tifffile, or rawpy → RGB in sRGB).
2. **Normalize linear data** (Smart16): percentile stretch to [0,1].
   - `--tiff-smart16` enables; `--tiff-smart16-pct lo,hi` choose percentiles.
   - `--tiff-smart16-perchannel` optionally stretches each RGB channel independently (more punch, small hue risk). If omitted, a **global curve** is used (safer color).
3. **Exposure** (scene‑linear): auto EV (optional, per‑image) and/or manual EV.
4. **Tonemap** (optional, scene‑linear → [0,1]): `none|reinhard|aces`.
5. **Gamma** (linear → display): e.g., 2.2.
6. **Display‑space shaping** (optional): black/white point mapping, contrast S‑curve, saturation.
7. **SSIM‑guided quality search** → save JPEG; attach EXIF/ICC; copy sidecars; optionally exiftool all‑tags.

---

## Quick start
Minimal “good defaults” for mixed TIFF/RAW with HDRish content:
```bash
python main.py "/Photos/Camera Roll" \
  --types tif,tiff,dng,jpg \
  --tiff-reader tifffile \
  --tiff-smart16 --tiff-smart16-pct 0.5,99.0 \
  --tiff-float-tonemap aces --tiff-gamma 2.2 \
  --auto-ev-mode mid_guard --auto-ev-mid 0.18 --auto-ev-mid-pct 50 \
  --auto-ev-hi-pct 99 --auto-ev-hi-cap 0.90 \
  --blackpoint-pct 0.3 --whitepoint-pct 99.7 \
  --contrast 0.12 --saturation 1.06 \
  --ssim 0.99 --ssim-downsample 4 --ssim-luma-only \
  --smart16-downsample 8 \
  --workers 6 --resume
```
**Tip:** If the whole set trends a bit dark/bright, add a global bias: `--tiff-exposure-ev ±0.2` (small nudge) or ±0.5.

---

## Avoiding the “gray veil”
If results look washed‑out:
1. Ensure **post‑gamma black/white mapping** is set (typical: `--blackpoint-pct 0.3 --whitepoint-pct 99.7`).
2. Don’t double‑tone‑map. Use **one** of: EV+gamma **and** (optionally) `--tiff-float-tonemap`.
3. If skies clip, keep `--auto-ev-mode mid_guard` with `--auto-ev-hi-cap 0.90`.
4. For punchier color at small risk of hue shifts, try `--tiff-smart16-perchannel`.

---

## Recipes
**A. Global look with highlight protection (recommended baseline)**
```bash
--tiff-smart16 --tiff-smart16-pct 0.5,99.0 \
--tiff-float-tonemap aces --tiff-gamma 2.2 \
--auto-ev-mode mid_guard --auto-ev-mid 0.18 --auto-ev-mid-pct 50 \
--auto-ev-hi-pct 99 --auto-ev-hi-cap 0.90 \
--blackpoint-pct 0.3 --whitepoint-pct 99.7 --contrast 0.12 --saturation 1.06
```

**B. When the set is too dark overall**
```bash
... baseline above ... --tiff-exposure-ev 0.3
```

**C. When skies are too hot**
- Lower `--auto-ev-hi-cap` (e.g., 0.85), or increase `--auto-ev-hi-pct` to target rarer highlights.

**D. Max speed for giant folders**
```bash
--no-brisque --ssim-downsample 4 --ssim-luma-only --smart16-downsample 8 --workers <N>
```

**E. Manual only (no per‑image auto)**
```bash
--auto-ev-mode off --tiff-exposure-ev 2.4 --tiff-float-tonemap aces --tiff-gamma 2.2 \
--blackpoint-pct 0.3 --whitepoint-pct 99.7
```

---

## Debugging & what to look for
Run with `--debug` (or `--debug-json`) to print per‑file mapping:
- Loader path, dtype/shape, photometric, bits/sample.
- Source min/max; percentile **lo/hi** values actually used.
- Auto‑EV internals (mode, mid/hi EVs, final EV).
- Luminance (linear) median **pre/post** tonemap.
- Output min/max after display shaping.

**Healthy signs**
- After post‑gamma mapping, printed `out_min≈0` and `out_max≈1` (or close).
- With `mid_guard`, highlight percentile stays ≤ `--auto-ev-hi-cap`.

---

## Performance notes
- **Workers**: start with ½ your logical CPUs. JPEG encoding tends to be CPU‑bound; SSIM search adds extra encodes.
- **Downsample knobs**: `--ssim-downsample 4` and `--smart16-downsample 8` are often visually indistinguishable but much faster.
- **Luma SSIM**: `--ssim-luma-only` is a strong speed win with little practical loss.
- **Disable BRISQUE** unless you need the score: `--no-brisque`.
- **EXIF copy**: If you don’t need full tag copies, use `--exiftool-mode none`.

---

## CLI (key options)
- Structure: `--flat`, `--resume`, `--types ext1,ext2`, `--workers N`, `--no-progress`
- TIFF/HDR: `--tiff-reader {auto|pillow|tifffile}`, `--tiff-smart16`, `--tiff-smart16-pct lo,hi`, `--tiff-smart16-perchannel`, `--tiff-float-tonemap {none|reinhard|aces}`, `--tiff-gamma G`, `--tiff-exposure-ev EV`, `--tiff-apply-icc`
- Auto‑EV: `--auto-ev-mode {off|mid|mid_guard}`, `--auto-ev-mid X`, `--auto-ev-mid-pct P`, `--auto-ev-hi-pct P`, `--auto-ev-hi-cap C`, `--auto-ev-downsample N`, `--auto-ev-bounds lo,hi`, `--auto-ev-iters I`
- Post‑gamma shaping: `--blackpoint-pct B`, `--whitepoint-pct W`, `--contrast C`, `--saturation S`
- Quality search: `--ssim THR`, `--qmin`, `--qmax`, `--progressive`, `--subsampling {0|1|2}`, `--ssim-downsample N`, `--ssim-luma-only`, `--search-optimize`
- Quality metrics: `--no-brisque`, `--brisque-model`, `--brisque-range`
- Metadata: `--exiftool-mode {all|none}`; sidecars copied automatically
- RAW: `--demosaic {AHD|LINEAR|AMAZE}`
- Diagnostics: `--debug`, `--debug-json`

---

## Troubleshooting
- **Looks flat/gray** → Add or tighten `--blackpoint-pct / --whitepoint-pct` (e.g., `0.3, 99.7`). Ensure only one tonemap step; avoid stacking conflicting curves.
- **Too bright overall** → Reduce global EV (`--tiff-exposure-ev -0.3`) or lower `--auto-ev-mid` (e.g., 0.16). With `mid_guard`, also lower `--auto-ev-hi-cap`.
- **Skies blown** → `--auto-ev-mode mid_guard` with `--auto-ev-hi-cap 0.85–0.90`.
- **Color shifts** → Turn off `--tiff-smart16-perchannel`; use global stretch. Ensure `--tiff-apply-icc` on Pillow path.
- **Slow** → Raise `--ssim-downsample`, enable `--ssim-luma-only`, raise `--smart16-downsample`, increase `--workers`, use `--no-brisque`, set `--exiftool-mode none`.

---

## Install
```bash
pip install pillow pillow-heif rawpy tifffile scikit-image tqdm opencv-contrib-python
brew install exiftool
```
Optional env:
```
BRISQUE_MODEL=/path/to/brisque_model_live.yml
BRISQUE_RANGE=/path/to/brisque_range_live.yml
```

---

## Summary
qJPEG now converts mixed HDRish TIFF/RAW sets into clean SDR JPEGs **without the gray veil**, balances exposure per image with **Auto‑EV mid_guard**, protects highlights, and lets you globally nudge exposure. With SSIM‑guided quality search, parallelism, and robust metadata carry‑over, it’s ready for large‑scale library runs.

