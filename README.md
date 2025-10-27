<div align="center">

# qJPEG — Intelligent JPEG Optimizer & Converter

**Perceptually-aware batch optimizer for HDR/RAW → SDR JPEG conversion**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Quick Start](QUICK_START.md) • [Implementation Plan](IMPLEMENTATION_PLAN.md) • [Website](https://qStivi.com)

</div>

---

## ✨ Overview

**qJPEG** is a perceptually-aware batch optimizer that intelligently converts high-bit-depth TIFF/RAW/HDR images to optimized JPEGs while preserving quality and metadata.

**Key Features:**
- 🎯 **SSIM-Guided Quality** - Finds the lowest JPEG quality meeting your target SSIM (structural similarity)
- 🌈 **HDR → SDR Conversion** - Smart tone mapping (ACES, Reinhard) with auto-exposure and highlight protection
- ⚡ **Fast Batch Processing** - Multiprocessing support with progress bars and resume capability
- 🎨 **No Gray Veil** - Intelligent black/white point mapping eliminates washed-out appearance
- 📋 **Config Presets** - Use simple presets instead of 20+ parameter commands
- 🔍 **Interactive Calibration** - Auto-tune settings for your specific images
- 📊 **Moiré Detection** - FFT-based detection flags unreliable quality scores on screen photographs
- 🗂️ **Metadata Preservation** - Copies EXIF, ICC profiles, and sidecars automatically

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install pillow pillow-heif rawpy tifffile scikit-image tqdm opencv-contrib-python

# Install exiftool (macOS/Linux)
brew install exiftool  # macOS
apt install libimage-exiftool-perl  # Linux

# Clone repository
git clone https://github.com/qStivi/qJPEG
cd qJPEG
```

### Basic Usage

```bash
# Use a config preset (recommended)
python main.py "/path/to/photos" --config hdr-default

# Auto-calibrate settings for your images
python calibrate.py "/path/to/photos" --output my-settings.yaml

# Use your custom settings
python main.py "/path/to/photos" --config my-settings
```

**[→ See QUICK_START.md for detailed guide](QUICK_START.md)**

---

## 🎯 NEW in v2.0: Simplified Workflow

**No more 20+ parameter commands!** Three new features make qJPEG easy to use:

### 1. 📋 Config File System
Use presets instead of long command lines:
```bash
python main.py "/path/to/photos" --config hdr-default
```

**Available presets:**
- `hdr-default` - Balanced HDR→SDR conversion with highlight protection
- `float32` - Aggressive clipping for 32-bit float TIFFs
- `camera-raw` - Conservative settings for camera RAW files
- `quick` - Fast processing for testing/iteration

### 2. 🎛️ Interactive Calibration
Automatically find optimal settings for your images:
```bash
python calibrate.py "/path/to/photos" --output my-settings.yaml
```

The calibration tool:
- Selects sample images from your library
- Processes them with current settings
- Asks questions about the results (too dark? skies blown out?)
- Automatically adjusts parameters based on your feedback
- Saves optimized preset for production runs

### 3. 🔍 Moiré Pattern Detection
Detects screen photographs and flags unreliable quality scores:
```bash
[OK] screen_photo.jpg -> output.jpg  quality=85, SSIM=0.9900, BRISQUE=52.34 (⚠️ moiré)
```

Uses FFT-based frequency analysis to identify moiré patterns with 100% accuracy on test sets.

---

## 🌟 Key Features Explained

<details>
<summary><b>🎨 HDR & Tone Mapping</b></summary>

- Percentile-based **Smart16** works for both **uint16 and float32/float64** TIFFs
- Optional **tone mapping** for float/linear content: `--tiff-float-tonemap {none|reinhard|aces}`
- **Exposure & Gamma** controls: `--tiff-exposure-ev`, `--tiff-gamma`
- **Post-gamma shaping** (display space):
  - `--blackpoint-pct` and `--whitepoint-pct` map luminance percentiles to 0/1 → kills "gray veil"
  - `--contrast` gentle S-curve
  - `--saturation` multiplier in display space

</details>

<details>
<summary><b>⚡ Auto-EV (Per-Image Exposure)</b></summary>

Automatic exposure adjustment for each image:

- `--auto-ev-mode {off|mid|mid_guard}`
  - **mid**: matches mid-tone luminance target (default 0.18) at chosen percentile
  - **mid_guard**: caps highlights to prevent blown skies (great for landscapes)
- Tunables: `--auto-ev-downsample`, `--auto-ev-bounds`, `--auto-ev-iters`
- Works with manual global offset (`--tiff-exposure-ev`) for set-wide bias

</details>

<details>
<summary><b>🚀 Performance Optimizations</b></summary>

- **SSIM search** can be downsampled and/or computed on luma only:
  - `--ssim-downsample N` (e.g., 4 → use every 4th pixel)
  - `--ssim-luma-only` - strong speed win with little quality loss
- **Smart16** percentile estimation can be subsampled: `--smart16-downsample N`
- **Multiprocessing**: `--workers N` - parallel processing with progress bars
- **Resume capability**: `--resume` - skip already processed files
- **Optional BRISQUE**: `--no-brisque` for max speed

</details>

<details>
<summary><b>🎯 Quality Control</b></summary>

- **SSIM-guided** quality search finds lowest quality meeting perceptual threshold
- **BRISQUE scores** (optional) for quality validation
- **Moiré detection** flags unreliable scores on screen photographs
- **Metadata preservation**: EXIF, ICC profiles, XMP sidecars

</details>

---

## 📖 Processing Pipeline

```
1. Load (Pillow/tifffile/rawpy → RGB in sRGB)
2. Normalize linear data (Smart16): percentile stretch to [0,1]
3. Exposure (scene-linear): auto EV + manual EV
4. Tonemap (optional): none|reinhard|aces
5. Gamma (linear → display): e.g., 2.2
6. Display-space shaping: black/white point, contrast, saturation
7. SSIM-guided quality search → save JPEG + metadata
```

---

## 🎨 Avoiding the "Gray Veil"

If results look washed-out:

1. ✅ Ensure **post-gamma black/white mapping** is set (typical: `--blackpoint-pct 0.3 --whitepoint-pct 99.7`)
2. ✅ Don't double-tone-map. Use **one** of: EV+gamma **and** (optionally) `--tiff-float-tonemap`
3. ✅ If skies clip, use `--auto-ev-mode mid_guard` with `--auto-ev-hi-cap 0.90`
4. ✅ For punchier color (small hue shift risk), try `--tiff-smart16-perchannel`

**For 32-bit float TIFFs**, use higher percentiles:
```bash
--blackpoint-pct 3.0 --whitepoint-pct 97.0
```

---

## 📋 Usage Examples

<details>
<summary><b>Recommended: Use Config Presets</b></summary>

```bash
# HDR/RAW with highlight protection
python main.py "/Photos" --config hdr-default

# 32-bit float TIFFs (aggressive clipping)
python main.py "/Photos" --config float32

# Override specific parameters
python main.py "/Photos" --config hdr-default --shadows 0.20 --workers 8
```

</details>

<details>
<summary><b>Advanced: Manual Configuration</b></summary>

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

**Tip:** Use calibration tool instead of manually tuning!

</details>

<details>
<summary><b>Common Scenarios</b></summary>

**Entire set too dark:**
```bash
python main.py "/Photos" --config hdr-default --tiff-exposure-ev 0.3
```

**Skies too hot:**
```bash
python main.py "/Photos" --config hdr-default --auto-ev-hi-cap 0.85
```

**Max speed for large batches:**
```bash
python main.py "/Photos" --config quick --workers 16 --no-brisque --exiftool-mode none
```

</details>

---

## 🔧 Command-Line Interface

<details>
<summary><b>Click to expand CLI options</b></summary>

### Structure
- `--flat` - Flat output (no directory structure)
- `--resume` - Skip already processed files
- `--types ext1,ext2` - File extensions to process
- `--workers N` - Parallel workers
- `--no-progress` - Disable progress bar

### TIFF/HDR Processing
- `--tiff-reader {auto|pillow|tifffile}` - TIFF loader
- `--tiff-smart16` - Enable Smart16 percentile stretching
- `--tiff-smart16-pct lo,hi` - Percentiles (default: 0.5,99.0)
- `--tiff-smart16-perchannel` - Per-channel stretch (punchier colors, hue shift risk)
- `--tiff-float-tonemap {none|reinhard|aces}` - Tone mapping operator
- `--tiff-gamma G` - Gamma correction (default: 2.2)
- `--tiff-exposure-ev EV` - Manual exposure adjustment
- `--tiff-apply-icc` - Apply embedded ICC profiles

### Auto-Exposure
- `--auto-ev-mode {off|mid|mid_guard}` - Auto-exposure mode
- `--auto-ev-mid X` - Mid-tone target luminance (default: 0.18)
- `--auto-ev-mid-pct P` - Mid-tone percentile (default: 50)
- `--auto-ev-hi-pct P` - Highlight percentile (default: 99)
- `--auto-ev-hi-cap C` - Highlight ceiling (default: 0.90)
- `--auto-ev-downsample N` - Downsample for speed
- `--auto-ev-bounds lo,hi` - EV search bounds
- `--auto-ev-iters I` - Max iterations

### Display Shaping
- `--blackpoint-pct B` - Black point percentile
- `--whitepoint-pct W` - White point percentile
- `--contrast C` - S-curve strength (0-0.5)
- `--saturation S` - Saturation multiplier

### Quality Search
- `--ssim THR` - Target SSIM threshold (default: 0.99)
- `--qmin` - Minimum JPEG quality (default: 1)
- `--qmax` - Maximum JPEG quality (default: 100)
- `--progressive` - Use progressive JPEG
- `--subsampling {0|1|2}` - Chroma subsampling
- `--ssim-downsample N` - Downsample SSIM computation
- `--ssim-luma-only` - Compute SSIM on luma only
- `--search-optimize` - Use Pillow optimize during search

### Quality Metrics
- `--no-brisque` - Disable BRISQUE scoring
- `--brisque-model PATH` - BRISQUE model file
- `--brisque-range PATH` - BRISQUE range file

### Metadata
- `--exiftool-mode {all|none}` - Metadata copy mode (default: all)

### RAW Processing
- `--demosaic {AHD|LINEAR|AMAZE}` - RAW demosaic algorithm

### Diagnostics
- `--debug` - Print per-file debug info
- `--debug-json` - Debug in JSON format

</details>

---

## 🐛 Troubleshooting

<details>
<summary><b>Common Issues</b></summary>

**Looks flat/gray:**
- Add or tighten `--blackpoint-pct / --whitepoint-pct` (e.g., `0.3, 99.7`)
- Ensure only one tonemap step; avoid stacking conflicting curves

**Too bright overall:**
- Reduce global EV (`--tiff-exposure-ev -0.3`)
- Lower `--auto-ev-mid` (e.g., 0.16)
- With `mid_guard`, also lower `--auto-ev-hi-cap`

**Skies blown:**
- Use `--auto-ev-mode mid_guard` with `--auto-ev-hi-cap 0.85-0.90`

**Color shifts:**
- Turn off `--tiff-smart16-perchannel`; use global stretch
- Ensure `--tiff-apply-icc` on Pillow path

**Slow processing:**
- Raise `--ssim-downsample` (e.g., 4)
- Enable `--ssim-luma-only`
- Raise `--smart16-downsample` (e.g., 8)
- Increase `--workers`
- Use `--no-brisque`
- Set `--exiftool-mode none`

</details>

---

## 🧪 Performance Notes

- **Workers**: Start with ½ your logical CPUs. JPEG encoding is CPU-bound
- **Downsample knobs**: `--ssim-downsample 4` and `--smart16-downsample 8` are often visually indistinguishable but much faster
- **Luma SSIM**: `--ssim-luma-only` is a strong speed win with little practical loss
- **Disable BRISQUE**: Unless you need the score, use `--no-brisque`
- **EXIF copy**: If you don't need full tag copies, use `--exiftool-mode none`

---

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Simple guide for common use cases
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Feature specifications and roadmap
- **[GRAY_VEIL_DEBUG_JOURNEY.md](GRAY_VEIL_DEBUG_JOURNEY.md)** - Deep dive into fixing washed-out images
- **[SESSION_STATE.md](SESSION_STATE.md)** - Current development state and context

---

## 🔬 Technical Features

### Moiré Pattern Detection (NEW)
Uses FFT-based frequency analysis to detect screen photographs:
- **Algorithm**: 2D FFT → magnitude spectrum → peak detection → confidence calculation
- **Accuracy**: 100% detection on test sets (58-60% confidence on screen photos, 0% on natural photos)
- **Performance**: ~5-10ms overhead per image
- **Output**: Flags unreliable BRISQUE scores with ⚠️ moiré warning

### Smart16 Processing
Intelligent percentile-based stretching for high-bit-depth images:
- Works with both **uint16** and **float32/float64** TIFFs
- Optional per-channel stretching for punchier colors
- Configurable percentiles for different image types

### Auto-Exposure Modes
- **mid**: Matches mid-tone luminance to photographic standard (18% gray)
- **mid_guard**: Adds highlight protection to prevent blown skies
- Per-image adaptive exposure for mixed lighting conditions

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Credits

Developed by [qStivi](https://qStivi.com)

**Dependencies:**
- [Pillow](https://python-pillow.org/) - Image processing
- [rawpy](https://github.com/letmaik/rawpy) - RAW file support
- [scikit-image](https://scikit-image.org/) - SSIM computation
- [OpenCV](https://opencv.org/) - BRISQUE quality assessment
- [ExifTool](https://exiftool.org/) - Metadata handling

---

<div align="center">

**[qStivi.com](https://qStivi.com)** • **[GitHub](https://github.com/qStivi/qJPEG)**

Made with ❤️ for photographers and image enthusiasts

</div>
