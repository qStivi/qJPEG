# qJPEG Development Session State

**Last Updated:** 2025-10-27
**Purpose:** Track development progress and capture conversation context for cross-machine continuity

---

## Quick Summary

### ✅ Completed Features
- **Moiré Pattern Detection** - FFT-based detection integrated into BRISQUE scoring pipeline

### 🔄 In Progress
- **BRISQUE-Driven Quality Control** - Design phase for inverse-adaptive compression

### 🎯 Next Steps
1. Download BRISQUE model files (required for testing)
2. Finalize design decisions for Pareto optimization approach
3. Implement BRISQUE-driven quality search
4. Test on pristine RAWs vs already-compressed JPEGs

---

## Session History

### Session 1-2: Initial Setup & Moiré Detection (2025-10-26)

**Objective:** Enhance BRISQUE quality scoring reliability for screen photographs

**Problem Identified:**
User noted that BRISQUE scores can be fooled by moiré patterns when photographing screens. Moiré patterns (aliasing artifacts from photographing fine grid patterns) confuse perceptual quality metrics like BRISQUE, producing unreliable scores.

**Solution Implemented:**
FFT-based moiré pattern detection with multi-scale validation

**Files Modified:**
1. `qjpeg/quality.py`:
   - Added `detect_moire_fft()` - FFT frequency analysis for moiré detection
   - Added `brisque_score_multiscale()` - Multi-resolution BRISQUE validation
   - Added `brisque_score_with_moire_check()` - Wrapper function with moiré detection

2. `qjpeg/pipeline.py`:
   - Integrated moiré detection into BRISQUE scoring pipeline
   - Added unreliable flag and moiré info to result dictionary
   - Display "⚠️ moiré" warning in output when patterns detected

3. `IMPLEMENTATION_PLAN.md`:
   - Documented Feature 3: Moiré Pattern Detection
   - Added comprehensive implementation details, testing results, and technical insights

**Technical Details:**
- **Algorithm:** 2D FFT → magnitude spectrum → peak detection → confidence calculation
- **Thresholds:** min_peak_ratio=1.5x, confidence=0.10 (10%)
- **Performance:** ~5-10ms overhead per image
- **Dependencies:** scipy (already a dependency)

**Testing Results:**
- ✅ Screen photos: 58-60% confidence, 10,000+ peaks - 100% detection rate
- ✅ Natural photos: 0% confidence, 0 peaks - 0% false negatives
- ⚠️ Digital artwork: 32.6% confidence, 10 peaks - Some false positives (acceptable for use case)

**Commits:**
- `430ce75` - Add FFT-based moiré pattern detection for BRISQUE quality scores
- `5661fcc` - Add multi-scale BRISQUE validation for moiré verification
- `a65c3bc` - Integrate moiré detection into BRISQUE scoring pipeline
- `124a6aa` - Document moiré detection implementation in IMPLEMENTATION_PLAN.md
- `9436667` - Add moiré detection test script for manual validation

---

### Session 3: BRISQUE-Driven Quality Control (2025-10-27)

**Objective:** Make BRISQUE actively drive compression decisions, not just report scores

**Current State:**
- BRISQUE is calculated but only used for informational display
- User wants to use BRISQUE before/after scores to guide compression
- Two goals: (1) Prevent degradation, (2) Enable MORE compression when safe

**Key User Insight - Inverse-Adaptive Compression:**

Traditional thinking (WRONG):
- Pristine source (BRISQUE=15) → strict protection, small delta allowed (≤5)
- Degraded source (BRISQUE=45) → can compress more, large delta allowed (≤15)

**User's corrected insight (RIGHT):**
- **Pristine RAW (BRISQUE=15)** → Can compress MUCH more (delta ≤20 acceptable)
  - *Rationale:* No existing artifacts, has "quality budget" to spend
  - *Benefit:* Much smaller files without perceptual quality loss

- **Already-compressed JPEG (BRISQUE=45)** → Must be conservative (delta ≤5)
  - *Rationale:* Already has artifacts (WhatsApp, previous processing)
  - *Benefit:* Don't compound artifacts on top of artifacts

**Why This Is Revolutionary:**
Pristine sources have MORE headroom for compression because they have no artifacts to compound. Already-degraded sources have LESS headroom because any additional compression adds artifacts on top of existing artifacts, causing perceptual quality to degrade rapidly.

**Example:**
- RAW photo (BRISQUE=18): Compress to Q=65, becomes BRISQUE=35 (+17) → Still looks excellent!
- WhatsApp JPEG (BRISQUE=48): Compress to Q=75, becomes BRISQUE=56 (+8) → Looks terrible!

**User's Chosen Approach:**
- **Optimization:** Pareto optimization balancing file size, SSIM, and BRISQUE delta
- **Moiré handling:** Fall back to SSIM-only when moiré detected
- **Thresholds:** Inverse-adaptive (pristine gets larger delta allowance, degraded gets smaller)

---

## Design Decisions Needed

Before implementing, need to finalize:

### 1. Pareto Scoring Function

**Option A: Simple weighted sum**
```python
score = (
    w_size * (1 - normalized_file_size) +      # Smaller is better
    w_ssim * normalized_ssim +                  # Higher is better
    w_brisque * (1 - normalized_brisque_delta)  # Lower delta is better
)
```

**Option B: Constraint satisfaction with size minimization**
```python
# Must satisfy BOTH constraints
if ssim >= ssim_min AND brisque_delta <= max_delta:
    # Among valid options, pick smallest file
    score = file_size  # Minimize this
```

### 2. Inverse-Adaptive Threshold Formula

**Option A: Linear inverse mapping**
```python
max_delta = max(5, min(20, 30 - (src_brisque / 2)))
```

**Option B: Tiered approach**
```python
if src_brisque < 20:      # Pristine (RAW, uncompressed)
    max_delta = 20
elif src_brisque < 35:    # Good (lightly compressed)
    max_delta = 12
elif src_brisque < 50:    # Fair (moderately compressed)
    max_delta = 7
else:                      # Poor (heavily compressed)
    max_delta = 3
```

**Option C: Exponential decay**
```python
max_delta = 20 * exp(-0.05 * src_brisque)
```

### 3. SSIM Floor When Using BRISQUE

**Option A: SSIM is absolute floor**
```python
# Never go below user's SSIM threshold
if ssim >= 0.99 AND brisque_delta <= max_delta:
    accept_quality()
```

**Option B: SSIM can be relaxed if BRISQUE is excellent**
```python
# If BRISQUE delta is tiny, allow lower SSIM
if brisque_delta < 3:
    ssim_floor = 0.95
elif brisque_delta < 10:
    ssim_floor = 0.97
else:
    ssim_floor = 0.99
```

---

## Prerequisites for Testing

### BRISQUE Model Files (CRITICAL - Required)

Currently missing from project. Need to download:

```bash
cd /path/to/qJPEG

# Download BRISQUE model files from OpenCV repository
curl -O https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml
curl -O https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml
```

**Verification:**
```bash
ls -lh brisque*.yml
# Should show two ~8KB files
```

**Without these files:**
- BRISQUE scoring returns None
- Cannot test any BRISQUE-driven features
- Pipeline falls back to SSIM-only mode

---

## Implementation Plan (Draft)

### Phase 1: Core BRISQUE-Driven Search

**File:** `qjpeg/quality.py`
**New Function:** `brisque_pareto_search()`

```python
def brisque_pareto_search(
    img: Image.Image,
    arr: np.ndarray,
    brisque_model: str,
    brisque_range: str,
    ssim_threshold: float = 0.99,
    qmin: int = 1,
    qmax: int = 100,
    inverse_adaptive: bool = True,
    moire_detected: bool = False,
    ...
) -> Tuple[int, float, Dict]:
    """
    Find optimal JPEG quality using Pareto optimization.

    Balances:
    - File size (smaller is better)
    - SSIM (higher is better)
    - BRISQUE delta (lower is better)

    Uses inverse-adaptive thresholds:
    - Pristine sources (low BRISQUE) → allow large delta (compress more)
    - Degraded sources (high BRISQUE) → allow small delta (compress less)
    """
```

### Phase 2: Inverse-Adaptive Threshold Calculation

**File:** `qjpeg/quality.py`
**New Function:** `calculate_brisque_threshold()`

```python
def calculate_brisque_threshold(
    src_brisque: float,
    mode: str = 'tiered'  # 'linear', 'tiered', 'exponential'
) -> float:
    """
    Calculate max allowed BRISQUE delta based on source quality.

    Inverse relationship: better source → larger delta allowed
    """
```

### Phase 3: Pipeline Integration

**File:** `qjpeg/pipeline.py`
Modify `process_one()` to:
- Check if BRISQUE models available
- If moiré detected → use SSIM-only mode
- If no moiré → use BRISQUE-driven search
- Report both SSIM and BRISQUE deltas

### Phase 4: Command-Line Interface

**File:** `main.py`
New arguments:
- `--brisque-mode {off|validate|drive}`
  - `off`: Don't use BRISQUE (current behavior)
  - `validate`: Use for post-hoc validation only (from IMPLEMENTATION_PLAN)
  - `drive`: Use to actively drive quality search (NEW)
- `--brisque-threshold-mode {fixed|linear|tiered|exponential}`
- `--brisque-max-delta <float>` (for fixed mode)
- `--brisque-pareto-weights <size,ssim,brisque>` (e.g., "0.4,0.3,0.3")

---

## Testing Strategy

### Test Set 1: Pristine RAW Photos
**Expected:** Should compress to Q=60-70 while maintaining good quality
- Source: Camera RAW files (expected BRISQUE=10-20)
- Measure: Final file size vs SSIM-only mode
- Goal: 30-40% smaller files with BRISQUE delta < 20

### Test Set 2: Already-Compressed JPEGs
**Expected:** Should use conservative compression
- Source: WhatsApp images, social media downloads (expected BRISQUE=40-60)
- Measure: BRISQUE delta after recompression
- Goal: Delta < 5, minimal quality loss

### Test Set 3: Screen Photos (Moiré Present)
**Expected:** Should fall back to SSIM-only
- Source: Photos from "Camera Roll/2024/08" with screen moiré
- Measure: Verify moiré detection triggers fallback
- Goal: Same behavior as SSIM-only mode

### Test Set 4: Mixed Collection
**Expected:** Inverse-adaptive behavior across quality spectrum
- Source: Mix of RAW, JPEG, PNG, screenshots
- Measure: Correlation between source BRISQUE and compression ratio
- Goal: Lower source BRISQUE → higher compression ratio

---

## Open Questions

1. **Pareto weights:** What's the right balance between size/SSIM/BRISQUE?
   - Start with equal weights (0.33, 0.33, 0.33)?
   - User-tunable via command line?

2. **Threshold mode:** Which formula for inverse-adaptive thresholds?
   - Tiered is most predictable
   - Exponential is smoothest
   - Linear is simplest

3. **SSIM floor:** Should we ever allow SSIM < user threshold?
   - Conservative: Never go below (Option A)
   - Aggressive: Allow if BRISQUE delta is excellent (Option B)

4. **Performance:** How much slower is BRISQUE-driven search?
   - Need to measure on real images
   - May need caching or optimization

---

## Machine-Specific Setup

This section is for reference when setting up on a new machine.

### Python Environment

**Required packages** (already in requirements):
```bash
pip install pillow pillow-heif rawpy tifffile scikit-image tqdm opencv-contrib-python numpy scipy networkx imageio
```

**Verify OpenCV BRISQUE support:**
```python
import cv2
print(hasattr(cv2, "quality"))  # Should be True
print(hasattr(cv2.quality, "QualityBRISQUE_create"))  # Should be True
```

### External Tools

**ExifTool:**
- macOS: `brew install exiftool`
- Windows: `winget install exiftool.exiftool`
- Linux: `apt install libimage-exiftool-perl`

**Git Configuration:**
```bash
git config --global user.name "qStivi"
git config --global user.email "stephanglaue@outlook.com"
```

### Project Setup

```bash
# Clone repository
git clone https://github.com/qStivi/qJPEG
cd qJPEG

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install pillow pillow-heif rawpy tifffile scikit-image tqdm opencv-contrib-python

# Download BRISQUE models (CRITICAL!)
curl -O https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml
curl -O https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml

# Verify setup
python main.py --help
```

---

## How to Continue This Work

### On Same Machine (Windows)
- Context preserved in this conversation
- Local state in `C:\Users\steph\CLAUDE.md`

### On Different Machine
1. **Clone repository:**
   ```bash
   git clone https://github.com/qStivi/qJPEG
   cd qJPEG
   ```

2. **Read this file first:**
   ```bash
   cat SESSION_STATE.md  # You're reading it now!
   ```

3. **Download BRISQUE models** (see Prerequisites section above)

4. **Review design decisions needed** (see Design Decisions section)

5. **Continue from where we left off:**
   - Finalize Pareto optimization approach
   - Implement BRISQUE-driven search
   - Test on real images

### Key Files to Read

- `SESSION_STATE.md` (this file) - Overall context
- `IMPLEMENTATION_PLAN.md` - Feature specifications
- `qjpeg/quality.py` - Current quality metrics implementation
- `qjpeg/pipeline.py` - Processing pipeline
- `test_moire.py` - Moiré detection testing tool

---

## Current Git State

**Branch:** main
**Unpushed commits:** 5 (as of 2025-10-27)
- `430ce75` - Add FFT-based moiré pattern detection
- `5661fcc` - Add multi-scale BRISQUE validation
- `a65c3bc` - Integrate moiré detection into pipeline
- `124a6aa` - Document moiré detection in IMPLEMENTATION_PLAN
- `9436667` - Add moiré detection test script

**Status:** Clean (after this session's commits are pushed)

---

## Notes for Future Claude Sessions

If you're a future Claude instance reading this:

1. **The user's key insight** about inverse-adaptive compression is the foundation for BRISQUE-driven quality control - don't lose sight of it!

2. **Moiré detection is complete and working** - don't reimplement it, just use it to fall back to SSIM when BRISQUE is unreliable.

3. **BRISQUE models MUST be downloaded** before testing - this is a hard requirement.

4. **Design decisions are still open** - user needs to choose Pareto approach, threshold formula, and SSIM floor behavior.

5. **Test on real images** - the user has test sets available:
   - RAW photos: `C:\Users\steph\Pictures\Bilder Freunde\Svone Geburtstag 2024` (or wherever they move them)
   - Screen photos: `C:\Users\steph\Pictures\Camera Roll\2024\08` (Windows machine)
   - Mixed collection: Various folders

6. **Performance matters** - qJPEG is used for batch processing thousands of images, so any new feature needs to be fast or optional.

---

**END OF SESSION STATE DOCUMENT**
