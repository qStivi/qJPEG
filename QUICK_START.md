# qJPEG Quick Start Guide

## New Features (v2.0)

### 1. Shadow Lift for Dark Subjects
Brighten dark areas without blowing out highlights:
```bash
--shadows 0.15   # Typical range: 0.1 - 0.3
```

### 2. Config File System
No more ridiculous 20+ parameter commands!

**Using presets:**
```bash
# Use the HDR default preset
python main.py "/path/to/photos" --config hdr-default

# Use camera RAW preset
python main.py "/path/to/photos" --config camera-raw

# Quick/fast preset for testing
python main.py "/path/to/photos" --config quick

# Override specific parameters
python main.py "/path/to/photos" --config hdr-default --shadows 0.20 --contrast 0.10
```

**Available presets:**
- `float32.yaml` - **RECOMMENDED for 32-bit float TIFFs** - Aggressive black/white point clipping
- `hdr-default.yaml` - Balanced HDR→SDR conversion with highlight protection
- `camera-raw.yaml` - Conservative settings for camera RAW files
- `quick.yaml` - Fast processing for testing/iteration
- `manual.yaml` - Full manual control, no auto-exposure

**Creating your own preset:**
```bash
# Save current settings to a config file
python main.py "/path/to/photos" --config hdr-default --shadows 0.20 \
  --save-config my-preset.yaml

# Use your custom preset
python main.py "/path/to/photos" --config my-preset.yaml
```

### 3. Interactive Calibration Tool
Automatically find optimal settings for your specific images!

**Basic usage:**
```bash
python calibrate.py "/path/to/photos" --output my-optimized.yaml
```

**The calibration process:**
1. Selects 5 sample images from your library
2. Processes them with current settings
3. Opens the output directory for you to review
4. Asks questions like:
   - "Are skies too bright?"
   - "Are subjects too dark?"
   - "Are colors too saturated?"
5. Automatically adjusts parameters based on your feedback
6. Repeats until you're satisfied
7. Saves optimized preset

**Advanced options:**
```bash
# Start from a specific preset
python calibrate.py "/path/to/photos" --base-preset hdr-default --output tuned.yaml

# Process more samples for better coverage
python calibrate.py "/path/to/photos" --samples 10

# Quick mode (faster iteration, lower quality)
python calibrate.py "/path/to/photos" --quick --output quick-test.yaml

# Limit iterations
python calibrate.py "/path/to/photos" --max-iterations 3
```

## 32-bit Float TIFF Images (IMPORTANT!)

If your images are **32-bit float** (not 16-bit), the gray veil issue requires **more aggressive** black/white point clipping:

**Why:** 32-bit float images after tone mapping often occupy a narrow range like [0.03, 0.93] instead of [0.0, 1.0]. Using very low percentiles like `0.2%` picks values too close to the min/max, so the remapping doesn't stretch enough.

**Solution:** Use **higher percentiles** to actually clip and stretch:

```bash
# Use the float32 preset (recommended)
python main.py "/path/to/32bit-float" --config float32

# Or manually specify aggressive clipping
python main.py "/path/to/32bit-float" --config hdr-default \
  --blackpoint-pct 3.0 --whitepoint-pct 97.0 --shadows 0.20
```

**Rule of thumb:**
- **16-bit images:** `blackpoint-pct 0.2-0.5`, `whitepoint-pct 99.5-99.8`
- **32-bit float:** `blackpoint-pct 2.0-5.0`, `whitepoint-pct 95.0-98.0`

**Check your image type:**
```python
import tifffile
with tifffile.TiffFile("your_image.tif") as tf:
    print(f"Bits: {tf.pages[0].bitspersample}")
    print(f"Format: {tf.pages[0].sampleformat}")  # 1=uint, 3=float
```

## Common Use Cases

### Case 1: Skies Too Bright, Subjects Too Dark
**Old way (painful):**
```bash
python main.py "/photos" --types tif,tiff --tiff-reader tifffile \
  --tiff-smart16 --tiff-smart16-pct 0.5,99.0 --smart16-downsample 4 \
  --tiff-float-tonemap aces --tiff-gamma 2.2 \
  --auto-ev-mode mid_guard --auto-ev-mid 0.18 --auto-ev-mid-pct 50 \
  --auto-ev-hi-pct 98.0 --auto-ev-hi-cap 0.90 --auto-ev-downsample 4 \
  --blackpoint-pct 0.2 --whitepoint-pct 99.7 \
  --contrast 0.06 --saturation 1.06 --shadows 0.0 \
  --ssim 0.99 --ssim-downsample 4 --ssim-luma-only \
  --no-brisque --exiftool-mode none --workers 8
```

**New way (simple):**
```bash
# Use calibration tool - it figures it out for you!
python calibrate.py "/photos" --output my-settings.yaml

# Or manually tweak
python main.py "/photos" --config hdr-default --shadows 0.20
```

### Case 2: Testing Different Settings Quickly
```bash
# Use quick preset for fast iteration
python main.py "/photos" --config quick --shadows 0.15

# Try different shadow lift values
python main.py "/photos" --config quick --shadows 0.10
python main.py "/photos" --config quick --shadows 0.20
python main.py "/photos" --config quick --shadows 0.25
```

### Case 3: Production Run with Optimal Settings
```bash
# Once you've found good settings with calibration:
python calibrate.py "/photos" --output production.yaml

# Then run on full library with high quality
python main.py "/full/library" --config production --workers 16 --resume
```

## Parameter Quick Reference

### Shadow Lift (`--shadows`)
- **What it does**: Brightens dark areas without affecting highlights
- **Range**: 0.0 (no lift) to 0.5 (strong lift)
- **Typical**: 0.10 - 0.25
- **Symptom**: Subjects too dark despite good sky exposure
- **Fix**: Increase shadows (e.g., `--shadows 0.20`)

### Auto-Exposure Mid Target (`--auto-ev-mid`)
- **What it does**: Target luminance for mid-tones (after tone mapping)
- **Range**: 0.05 to 0.50
- **Default**: 0.18 (18% gray, photographic standard)
- **Symptom**: Entire image too dark/bright
- **Fix**: Increase for brighter (e.g., 0.22), decrease for darker (e.g., 0.16)

### Highlight Cap (`--auto-ev-hi-cap`)
- **What it does**: Maximum allowed highlight luminance (protects skies)
- **Range**: 0.5 to 1.0
- **Default**: 0.90
- **Symptom**: Skies blown out
- **Fix**: Lower cap (e.g., 0.85 or 0.88)

### Black/White Point
- **What it does**: Maps luminance percentiles to pure black/white
- **Defaults**: 0.2 / 99.7
- **Symptom**: Gray veil, washed out appearance
- **Fix**: Tighten range (e.g., `--blackpoint-pct 0.3 --whitepoint-pct 99.5`)

## Tips & Tricks

1. **Always specify both** `--blackpoint-pct` and `--whitepoint-pct` together, or use neither
2. **Use `--shadows` instead of increasing `--auto-ev-mid`** when highlights are already good
3. **Start with calibration tool** - saves hours of trial and error
4. **Use `--quick` preset** for testing, then switch to quality preset for final run
5. **Enable `--resume`** for large batches - you can stop/restart anytime
6. **Check `--debug-json` output** to understand what's happening under the hood

## Troubleshooting

### "Config file not found"
Make sure you're using just the preset name without path or extension:
```bash
# ✓ Correct
--config hdr-default

# ✗ Wrong
--config presets/hdr-default.yaml
```

### "PyYAML not installed"
Install it:
```bash
pip install pyyaml
```

### Calibration tool timeout
Use `--quick` mode or reduce `--samples`:
```bash
python calibrate.py "/photos" --quick --samples 3
```

### Results still not good
1. Try starting calibration from a different base preset
2. Check if you need per-channel stretching: `--tiff-smart16-perchannel`
3. Review the debug output to see actual values being applied
4. Some images may just be difficult - consider manual EV adjustment

## Examples

### Example 1: First Time User
```bash
# Let calibration tool figure it out
python calibrate.py ~/Pictures/CameraRoll --output my-camera.yaml

# Use your optimized settings
python main.py ~/Pictures/CameraRoll --config my-camera --workers 8 --resume
```

### Example 2: Known Underexposed Images
```bash
# Start with more shadow lift
python main.py ~/Photos --config hdr-default --shadows 0.25 --auto-ev-mid 0.22
```

### Example 3: Landscape Photography with Bright Skies
```bash
# More aggressive highlight protection
python main.py ~/Landscapes --config hdr-default \
  --auto-ev-hi-cap 0.85 --auto-ev-hi-pct 99.5
```

### Example 4: Studio/Indoor Photography (No Sky Issues)
```bash
# Can push mid-tones brighter without highlight guard
python main.py ~/Studio --config manual \
  --auto-ev-mode mid --auto-ev-mid 0.22
```
