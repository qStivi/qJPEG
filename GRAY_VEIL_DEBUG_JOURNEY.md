# The Gray Veil Mystery: A Debugging Journey

## TL;DR - The Fix

**Problem:** Images had a "gray veil" - blacks weren't truly black (0.03-0.07) and whites weren't truly white (0.92-0.95), making images look washed out like over-brightened video footage.

**Root Cause:** Shadow lift and contrast S-curve were applied **AFTER** black/white point remapping, re-adding the gray veil that was just removed.

**Solution:** Set `shadows: 0.0` and `contrast: 0.0` in all presets.

**Test Command:**
```bash
python main.py "/path/to/32bit-float" --config float32
```

Expected debug output:
```json
"out_min": 0.0,  // Perfect black
"out_max": 1.0   // Perfect white
```

---

## Timeline of Events

### Commit c30a75d (Oct 23, 2025) - The Original Fix
**"Add TIFF exposure/gamma correction, ICC profile support"**

User implemented gamma correction with this key insight:
> "arr01: float32 in [0,1]. Apply exposure (EV) then gamma (to sRGB-like display)."

The pipeline was simple and **worked perfectly**:
```
1. Percentile stretch (0.5%, 99.0%) → [0,1] linear
2. Auto-EV multiplication → still linear
3. ACES tonemap → compress highlights, still linear
4. Gamma correction (^1/2.2) → linear to display space
5. Convert to uint8 → Done!
```

**No gray veil** at this point.

### Commit eec94df (Later) - Post-Gamma Shaping Added
**"Add post-gamma shaping with black/white point remapping"**

Added new `_apply_post_gamma_shaping()` function with:
- Black/white point remapping (percentile-based)
- Contrast S-curve
- Saturation adjustment

The pipeline became:
```
1. Percentile stretch → [0,1] linear
2. Auto-EV → still linear
3. ACES tonemap → still linear
4. Gamma → display space
5. Black/white point remap → [0,1] display ← GOOD!
6. Shadow lift → lifts blacks from 0 to 0.01+ ← BAD!
7. Contrast S-curve → compresses to [0.07, 0.93] ← BAD!
8. Saturation → OK
9. Convert to uint8
```

Gray veil starts appearing intermittently.

### Commit 94f8189 (Optimization Commit) - Gray Veil Returns
**"Add SSIM, metadata, and Smart16 handling optimizations"**

Changed default for `TIFF_SMART16_PERCHANNEL` from True to False, and added "safety net" for black/white point defaults.

When user specified only `--blackpoint-pct 0.2` without `--whitepoint-pct`, it triggered:
- `blackpoint_pct = 0.2`
- `whitepoint_pct = 100.0` (default, no remapping!)

This left the upper end washed out.

### Oct 25, 2025 - The Investigation Begins

**User's Command (20+ parameters!):**
```bash
python main.py "/Users/qstivi/Pictures/Camera Roll" \
  --types tif,tiff --tiff-reader tifffile \
  --tiff-smart16 --tiff-smart16-pct 0.5,99.0 --smart16-downsample 4 \
  --tiff-float-tonemap aces --tiff-gamma 2.2 \
  --auto-ev-mode mid_guard --auto-ev-mid 0.28 --auto-ev-mid-pct 75 \
  --auto-ev-hi-pct 98.0 --auto-ev-hi-cap 0.94 --auto-ev-downsample 4 \
  --blackpoint-pct 0.2 \
  --contrast 0.06 --saturation 1.06 \
  --ssim 0.99 --ssim-downsample 4 --ssim-luma-only \
  --no-brisque --exiftool-mode none --workers 1 --debug-json
```

**Issues Identified:**
1. Missing `--whitepoint-pct` → no upper remapping
2. `--auto-ev-mid 0.28` → 56% too bright, but clamped by highlight guard
3. **32-bit float TIFFs** need different percentiles than 16-bit

**Debug Output Showed:**
```json
"out_min": 0.030296,  // Should be 0.0!
"out_max": 0.933686,  // Should be 1.0!
```

---

## The 32-Bit Float Discovery

User asked: **"Did you think about that my images are 32 bit? not 16 or 8?"**

This was the critical insight!

### Why 32-Bit Float is Different

**16-bit uint images:**
- Raw range: [0, 65535]
- After percentile stretch: uses nearly full [0,1] range
- 0.2 percentile ≈ 130 out of 65535
- Black/white point 0.2%/99.8% works fine

**32-bit float images:**
- Raw range: [-0.26, 16.0] (scene-linear, HDR)
- After ACES tonemap + gamma: [0.003, 0.965]
- 0.2 percentile ≈ 0.0033 (very close to min 0.003!)
- Remapping (val - 0.0033) / (0.965 - 0.0033) barely stretches anything

**The Fix:** Use higher percentiles for 32-bit float:
- 16-bit: `blackpoint_pct: 0.2-0.5%`, `whitepoint_pct: 99.5-99.8%`
- 32-bit float: `blackpoint_pct: 2.0-5.0%`, `whitepoint_pct: 95.0-98.0%`

---

## The Shadow Lift & Contrast Revelation

User's critical comment pointed to commit c30a75d:
> "I think the gamma AFTER part was VERY important."

This made me check **what came after** gamma in the current code vs. c30a75d.

**c30a75d:** Gamma → Done (no post-processing)
**Current:** Gamma → Black/white remap → **Shadow lift** → **Contrast S-curve** → Done

Testing with `--blackpoint-pct 0 --whitepoint-pct 100` (disabling remap) **still had gray veil**, proving the issue was in shadow lift or contrast.

### The Diagnostic Script Revealed All

Created `analyze_tiff.py` to simulate the entire pipeline. Results:

**After black/white point remap (3%/97%):**
```
Min: 0.000000, Max: 1.000000  ✓ Perfect!
```

**After shadow lift (0.20):**
```
Min: 0.001043, Max: 1.000000  ✗ Blacks lifted!
```

**After contrast S-curve (0.06):**
```
Min: 0.077272, Max: 0.922728  ✗ Whites compressed!
```

**THAT'S THE GRAY VEIL!**

### Mathematical Proof

**Shadow lift** adds based on darkness:
```python
mask = (1.0 - luminance)²
arr = arr + shadow_lift * mask

For black (0.0): mask = 1.0² = 1.0
                 result = 0.0 + 0.20 * 1.0 = 0.20  ← Gray!
```

**Contrast S-curve** compresses around midpoint:
```python
gain = 2.0 + 4.0 * contrast * 2.0
arr = 0.5 + tanh((arr - 0.5) * gain) * 0.5

For contrast=0.06: gain = 2.48
For white (1.0): tanh((1.0 - 0.5) * 2.48) = tanh(1.24) = 0.846
                 result = 0.5 + 0.846 * 0.5 = 0.923  ← Gray!
```

---

## The Final Test

User ran:
```bash
python main.py "/Users/qstivi/Pictures/Camera Roll" \
  --config float32 \
  --shadows 0.0 \
  --contrast 0.0 \
  --workers 1 --debug-json
```

**Result:**
```json
"out_min": 0.0,  ✓ PERFECT BLACK!
"out_max": 1.0,  ✓ PERFECT WHITE!
```

**User's reaction:** "OMG! YES That fixed it. YES!"

---

## Technical Deep Dive

### The Processing Pipeline

**Correct Pipeline (c30a75d):**
```
Raw float32 [-0.26, 16.0]
↓ Percentile stretch (0.5%, 99%)
[0,1] scene-linear
↓ Auto-EV (* 2^1.14)
[0,2.2] scene-linear
↓ ACES tonemap
[0,0.93] scene-linear (compressed)
↓ Gamma (^1/2.2)
[0,0.96] display space
↓ Black/white remap (3%/97%)
[0,1] display space ✓
↓ TO UINT8
[0,255]
```

**Broken Pipeline (with shadow lift + contrast):**
```
... (same as above until black/white remap)
[0,1] display space ✓
↓ Shadow lift
[0.001,1] display space ✗
↓ Contrast S-curve
[0.077,0.923] display space ✗✗
↓ TO UINT8
[20,235] instead of [0,255] → GRAY VEIL
```

### Why It Looked Like "Video Game Gamma Too High"

User's description:
> "The same effect as if I had my 'gamma' setting in a video game too high or when someone brightens video footage in a youtube video bc its so dark"

This is **exactly right**! When you:
1. Lift all values away from 0 (shadow lift)
2. Compress all values away from 1 (contrast)
3. The histogram moves from [0, 255] to [20, 235]

This is identical to:
- Video game "brightness" slider (lifts blacks)
- YouTube over-brightening (compresses to limited range)
- Incorrect display gamma (gamma 2.4 content on gamma 2.2 display)

---

## The Solution

### Code Changes

**All presets updated:**
```yaml
# Post-gamma shaping
blackpoint_pct: 2.0-3.0   # Higher for float32
whitepoint_pct: 97.0-98.0 # Lower for float32
shadows: 0.0              # DISABLED - causes gray veil
contrast: 0.0             # DISABLED - S-curve compresses range
saturation: 1.06          # OK - doesn't affect range
```

### Why Not Reorder Instead?

**Option A:** Apply shadow/contrast BEFORE black/white remap
```
Gamma → Shadow lift → Contrast → Black/white remap
```

**Problem:** Shadow lift and contrast work better in display space with known [0,1] range. Applying them before remapping means working with arbitrary [0.003, 0.965] range.

**Option B:** Disable shadow lift and contrast entirely ✓
```
Gamma → Black/white remap → Done
```

**This matches the working c30a75d pipeline!**

Future enhancement: If shadow lift is needed, implement it in **linear space** before gamma, not display space after.

---

## Lessons Learned

### 1. Image Bit Depth Matters

Don't assume 16-bit and 32-bit float behave the same:
- 16-bit: Nearly full-range after stretch
- 32-bit float: Scene-linear HDR, needs tonemap + aggressive percentile clipping

### 2. Order of Operations is Critical

```
Good: Percentile stretch → EV → Tonemap → Gamma → Black/white remap
Bad:  Percentile stretch → EV → Tonemap → Gamma → Black/white remap → Shadow lift → Contrast
```

Each operation after black/white remap can re-introduce the gray veil.

### 3. Percentile-Based Remapping Needs Meaningful Percentiles

For 32-bit float after tonemap:
- 0.2% percentile is too close to minimum (0.003 vs 0.0033)
- 3.0% percentile creates meaningful separation (0.020 vs 0.003)

### 4. Diagnostic Tools are Essential

The `analyze_tiff.py` script that simulated the entire pipeline was the key to finding the exact step where the veil was reintroduced.

### 5. Comments in Code Tell Stories

The comment "gamma (to sRGB-like display)" in c30a75d was the breadcrumb that led to the solution. **Document your insights!**

---

## How to Verify Your Images Don't Have Gray Veil

### Quick Visual Test
Open the JPEG in any image viewer. The histogram should:
- Touch 0 on the left (true black)
- Touch 255 on the right (true white)
- NOT have empty space at edges

### Command-Line Test
```bash
python main.py "/path/to/image" --config float32 --workers 1 --debug-json
```

Look for:
```json
"out_min": 0.0,   // Must be exactly 0
"out_max": 1.0    // Must be exactly 1
```

If you see `"out_min": 0.030` or `"out_max": 0.920`, you have a gray veil.

### Diagnostic Script
```bash
python analyze_tiff.py "/path/to/image.TIF"
```

Look for the recommendations at the end.

---

## Recommended Settings by Image Type

### 32-bit Float TIFF (Scene-Linear HDR)
```bash
--config float32
# Or manually:
--blackpoint-pct 3.0 --whitepoint-pct 97.0 \
--shadows 0.0 --contrast 0.0
```

### 16-bit TIFF (Linear or Gamma-Encoded)
```bash
--config hdr-default
# Or manually:
--blackpoint-pct 2.0 --whitepoint-pct 98.0 \
--shadows 0.0 --contrast 0.0
```

### Camera RAW (DNG, CR2, NEF, etc.)
```bash
--config camera-raw
# Or manually:
--blackpoint-pct 2.5 --whitepoint-pct 97.5 \
--shadows 0.0 --contrast 0.0
```

---

## Future Enhancements

### 1. Automatic Bit Depth Detection
```python
if bits_per_sample == 32 and sample_format == 3:  # float32
    blackpoint_pct = 3.0
    whitepoint_pct = 97.0
elif bits_per_sample == 16:
    blackpoint_pct = 0.5
    whitepoint_pct = 99.5
```

### 2. Shadow Lift in Linear Space
Apply shadow lift **before** gamma, in scene-linear space:
```python
# In linear space, before gamma
shadow_lift_linear = 0.1
arr_linear = arr_linear + shadow_lift_linear * mask
# Then apply gamma
arr_display = arr_linear ** (1/2.2)
```

This wouldn't interfere with black/white point remapping in display space.

### 3. Adaptive Percentile Selection
Analyze the histogram and automatically choose percentiles:
```python
# If 0.2% is very close to min, use higher percentile
if pct_0_2 - min_val < 0.01:
    blackpoint_pct = 3.0
```

### 4. Warning System
```python
if out_min > 0.01 or out_max < 0.99:
    print("[WARNING] Gray veil detected!")
    print(f"  Output range: [{out_min:.3f}, {out_max:.3f}]")
    print(f"  Recommendation: Adjust black/white point percentiles")
```

---

## References

- **c30a75d:** "Add TIFF exposure/gamma correction" - The working baseline
- **eec94df:** "Add post-gamma shaping" - Where shadow/contrast were introduced
- **94f8189:** "Add SSIM optimizations" - When gray veil returned
- **User's insight:** "Did you think about that my images are 32 bit?"
- **User's comment:** "I think the gamma AFTER part was VERY important"

---

## Acknowledgments

This debugging journey was a collaboration between:
- User's domain expertise (photography, 32-bit float TIFFs, visual perception)
- User's critical insights (c30a75d comment, 32-bit observation)
- Systematic debugging (debug JSON output, analyze_tiff.py)
- Git history analysis (finding what changed between working and broken)

The solution was hiding in plain sight: **Go back to the simplicity of c30a75d.**

---

**Date:** October 25, 2025
**Status:** ✅ RESOLVED
**Fix:** Set `shadows: 0.0` and `contrast: 0.0` in all presets
