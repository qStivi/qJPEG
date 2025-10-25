#!/usr/bin/env python3
"""
Interactive calibration tool for qJPEG
Helps you find optimal settings for your image library through visual feedback.

Usage:
    python calibrate.py "/path/to/Camera Roll" --samples 5 --output my-preset.yaml
    python calibrate.py "/path/to/Camera Roll" --base-preset hdr-default
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import json
from typing import Dict, Any, List, Optional
import shutil

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    print("[ERROR] PyYAML required for calibration tool: pip install pyyaml")
    sys.exit(1)

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    print("[ERROR] Pillow required: pip install pillow")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent

# Default parameter ranges for tuning
PARAM_ADJUSTMENTS = {
    # Format: param_name -> {feedback_type: adjustment_delta}
    "auto_ev_hi_cap": {
        "sky_too_bright": -0.02,
        "sky_too_dark": +0.02,
        "sky_good": 0.0,
    },
    "auto_ev_mid": {
        "subjects_too_dark": +0.02,
        "subjects_too_bright": -0.02,
        "subjects_good": 0.0,
    },
    "shadows": {
        "subjects_too_dark": +0.05,
        "subjects_too_bright": -0.05,
        "subjects_good": 0.0,
    },
    "contrast": {
        "too_flat": +0.02,
        "too_contrasty": -0.02,
        "contrast_good": 0.0,
    },
    "saturation": {
        "too_dull": +0.05,
        "too_saturated": -0.05,
        "saturation_good": 0.0,
    },
    "blackpoint_pct": {
        "too_washed": +1.0,        # Meaningful step for percentile (was 0.1, way too small!)
        "too_crushed": -1.0,
        "blacks_good": 0.0,
    },
    "whitepoint_pct": {
        "too_washed": -1.0,        # Meaningful step for percentile
        "highlights_clipped": +1.0,
        "whites_good": 0.0,
    }
}


class Calibrator:
    def __init__(self, input_root: Path, base_preset: Optional[str] = None,
                 samples: int = 5, quick_mode: bool = False):
        self.input_root = input_root
        self.samples = samples
        self.quick_mode = quick_mode
        self.temp_dir = SCRIPT_DIR / "calibration_temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Load base preset or use defaults
        if base_preset:
            self.params = self.load_preset(base_preset)
        else:
            self.params = self.get_default_params()

        # Override with quick settings if requested
        if quick_mode:
            self.params.update({
                "ssim_downsample": 8,
                "auto_ev_downsample": 16,
                "smart16_downsample": 8,
                "ssim": 0.95,
                "workers": 1,
            })

        self.sample_files = []
        self.iteration = 0

    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a preset YAML file."""
        preset_path = SCRIPT_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            preset_path = Path(preset_name)

        if not preset_path.exists():
            print(f"[WARNING] Preset not found: {preset_name}, using defaults")
            return self.get_default_params()

        with open(preset_path, 'r') as f:
            params = yaml.safe_load(f)
        print(f"[CONFIG] Loaded base preset: {preset_path}")
        return params if params else self.get_default_params()

    def get_default_params(self) -> Dict[str, Any]:
        """Get sensible default parameters."""
        return {
            "types": "tif,tiff",
            "tiff_reader": "tifffile",
            "tiff_smart16": True,
            "tiff_smart16_pct": [0.5, 99.0],
            "tiff_smart16_perchannel": True,
            "smart16_downsample": 4,
            "tiff_float_tonemap": "aces",
            "tiff_gamma": 2.2,
            "auto_ev_mode": "mid_guard",
            "auto_ev_mid": 0.18,
            "auto_ev_mid_pct": 50,
            "auto_ev_hi_pct": 98.0,
            "auto_ev_hi_cap": 0.90,
            "auto_ev_downsample": 4,
            "blackpoint_pct": 2.0,   # For 32-bit float images
            "whitepoint_pct": 98.0,  # For 32-bit float images
            "shadows": 0.15,
            "contrast": 0.06,
            "saturation": 1.06,
            "ssim": 0.99,
            "ssim_downsample": 4,
            "ssim_luma_only": True,
            "no_brisque": True,
            "exiftool_mode": "none",
            "workers": 1,
        }

    def find_sample_files(self):
        """Find sample image files from the input directory."""
        print(f"[SCAN] Looking for sample files in {self.input_root}")

        # Supported extensions
        exts = {".tif", ".tiff", ".dng", ".cr2", ".cr3", ".nef", ".arw"}

        # Find all matching files
        all_files = []
        for ext in exts:
            all_files.extend(self.input_root.rglob(f"*{ext}"))
            all_files.extend(self.input_root.rglob(f"*{ext.upper()}"))

        if not all_files:
            print(f"[ERROR] No supported image files found in {self.input_root}")
            return False

        # Sample evenly across the collection
        step = max(1, len(all_files) // self.samples)
        self.sample_files = all_files[::step][:self.samples]

        print(f"[SAMPLES] Selected {len(self.sample_files)} files:")
        for f in self.sample_files:
            print(f"  - {f.name}")

        return True

    def build_command(self, output_suffix: str = "") -> List[str]:
        """Build the main.py command with current parameters."""
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "main.py"),
            str(self.input_root),
            "--debug-json",
        ]

        # Add all parameters
        for key, value in self.params.items():
            # Skip None values
            if value is None:
                continue

            # Convert to CLI arg format
            cli_key = f"--{key.replace('_', '-')}"

            # Handle booleans
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_key)
            # Handle lists
            elif isinstance(value, list):
                cmd.append(cli_key)
                cmd.append(','.join(str(v) for v in value))
            # Handle regular values
            else:
                cmd.append(cli_key)
                cmd.append(str(value))

        return cmd

    def process_samples(self, output_suffix: str = "") -> bool:
        """Process sample files with current parameters."""
        print(f"\n[PROCESS] Running with current parameters (iteration {self.iteration})...")

        cmd = self.build_command(output_suffix)

        # Run main.py
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Look for debug JSON output
            debug_data = []
            for line in result.stdout.split('\n'):
                if line.strip().startswith('{'):
                    try:
                        debug_data.append(json.loads(line))
                    except:
                        pass

            if result.returncode != 0:
                print(f"[ERROR] Processing failed:")
                print(result.stderr)
                return False

            print(f"[OK] Processed {len(debug_data)} files")
            return True

        except subprocess.TimeoutExpired:
            print("[ERROR] Processing timed out (5 min limit)")
            return False
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            return False

    def show_results(self):
        """Show results and prompt for feedback."""
        output_dir = self.input_root.parent / f"{self.input_root.name}_compressed"

        print(f"\n[RESULTS] Output directory: {output_dir}")
        print("\nPlease review the processed images and provide feedback.")
        print("You can open the images in your favorite viewer.")

        # Try to open the output directory
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(output_dir)])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["explorer", str(output_dir)])
            elif sys.platform.startswith("linux"):  # Linux
                subprocess.run(["xdg-open", str(output_dir)])
        except:
            pass

    def get_feedback(self) -> Dict[str, bool]:
        """Prompt user for feedback on current results."""
        print("\n" + "="*60)
        print("FEEDBACK QUESTIONS")
        print("="*60)

        feedback = {}

        questions = [
            ("sky_too_bright", "Are skies/highlights too BRIGHT? (y/n): "),
            ("sky_too_dark", "Are skies/highlights too DARK? (y/n): "),
            ("subjects_too_dark", "Are subjects/mid-tones too DARK? (y/n): "),
            ("subjects_too_bright", "Are subjects/mid-tones too BRIGHT? (y/n): "),
            ("too_flat", "Does the image look too FLAT (low contrast)? (y/n): "),
            ("too_contrasty", "Does the image look too CONTRASTY? (y/n): "),
            ("too_washed", "Does the image look WASHED OUT (gray veil)? (y/n): "),
            ("too_dull", "Are colors too DULL? (y/n): "),
            ("too_saturated", "Are colors too SATURATED? (y/n): "),
        ]

        for key, question in questions:
            while True:
                response = input(question).strip().lower()
                if response in ['y', 'yes', 'n', 'no', '']:
                    feedback[key] = response in ['y', 'yes']
                    break
                print("Please answer 'y' or 'n'")

        return feedback

    def apply_feedback(self, feedback: Dict[str, bool]):
        """Adjust parameters based on user feedback."""
        print("\n[ADJUST] Applying feedback...")

        adjustments_made = []

        for param, feedback_map in PARAM_ADJUSTMENTS.items():
            for feedback_type, delta in feedback_map.items():
                if feedback.get(feedback_type, False) and delta != 0:
                    old_val = self.params.get(param, 0.0)
                    if isinstance(old_val, (int, float)):
                        new_val = old_val + delta
                        # Clamp values
                        if param == "auto_ev_hi_cap":
                            new_val = max(0.5, min(1.0, new_val))
                        elif param == "auto_ev_mid":
                            new_val = max(0.05, min(0.50, new_val))
                        elif param == "shadows":
                            new_val = max(0.0, min(0.5, new_val))
                        elif param == "contrast":
                            new_val = max(0.0, min(0.5, new_val))
                        elif param == "saturation":
                            new_val = max(0.5, min(2.0, new_val))
                        elif param.endswith("_pct"):
                            new_val = max(0.0, min(100.0, new_val))

                        self.params[param] = round(new_val, 3)
                        adjustments_made.append(f"  {param}: {old_val:.3f} → {new_val:.3f} ({feedback_type})")

        if adjustments_made:
            print("Adjustments made:")
            for adj in adjustments_made:
                print(adj)
        else:
            print("No adjustments needed - looks good!")

        return len(adjustments_made) > 0

    def save_preset(self, output_path: str):
        """Save current parameters to a preset file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, 'w') as f:
            yaml.dump(self.params, f, default_flow_style=False, sort_keys=False)

        print(f"\n[SAVED] Preset saved to: {p}")
        print(f"\nYou can now use this preset with:")
        print(f"  python main.py \"/path/to/photos\" --config {p}")

    def run_calibration(self, max_iterations: int = 5, output_preset: str = "calibrated.yaml"):
        """Run the interactive calibration loop."""
        print("="*60)
        print("qJPEG INTERACTIVE CALIBRATION")
        print("="*60)

        if not self.find_sample_files():
            return False

        # Initial parameters summary
        print("\n[PARAMS] Starting parameters:")
        for key, value in sorted(self.params.items()):
            if key not in ['types', 'workers', 'no_brisque', 'exiftool_mode']:
                print(f"  {key}: {value}")

        # Calibration loop
        for i in range(max_iterations):
            self.iteration = i + 1

            # Process samples
            if not self.process_samples():
                print("[ERROR] Processing failed, aborting calibration")
                return False

            # Show results and get feedback
            self.show_results()
            feedback = self.get_feedback()

            # Check if user is satisfied
            all_good = all(not v for k, v in feedback.items() if k.endswith('_good') is False)
            if all_good or self.iteration >= max_iterations:
                satisfied = input("\nAre you satisfied with the results? (y/n): ").strip().lower()
                if satisfied in ['y', 'yes']:
                    print("\n[SUCCESS] Calibration complete!")
                    self.save_preset(output_preset)
                    return True
                elif self.iteration >= max_iterations:
                    print(f"\n[DONE] Max iterations ({max_iterations}) reached.")
                    save_anyway = input("Save current settings anyway? (y/n): ").strip().lower()
                    if save_anyway in ['y', 'yes']:
                        self.save_preset(output_preset)
                    return False

            # Apply feedback and continue
            if not self.apply_feedback(feedback):
                print("\nNo adjustments made. Trying again...")

            print(f"\n[ITERATE] Iteration {self.iteration + 1}/{max_iterations}...")

        print("\n[DONE] Calibration finished")
        self.save_preset(output_preset)
        return True

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive calibration tool for qJPEG - find optimal settings through visual feedback"
    )
    parser.add_argument("input_root", type=str, help="Folder containing sample images")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of sample images to process (default: 5)")
    parser.add_argument("--base-preset", type=str, default=None,
                       help="Base preset to start from (e.g., 'hdr-default')")
    parser.add_argument("--output", type=str, default="calibrated.yaml",
                       help="Output preset filename (default: calibrated.yaml)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum calibration iterations (default: 5)")
    parser.add_argument("--quick", action="store_true",
                       help="Use quick/low-quality settings for faster iteration")

    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}")
        return 1

    calibrator = Calibrator(
        input_root=input_root,
        base_preset=args.base_preset,
        samples=args.samples,
        quick_mode=args.quick,
    )

    try:
        success = calibrator.run_calibration(
            max_iterations=args.max_iterations,
            output_preset=args.output,
        )
        return 0 if success else 1
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    sys.exit(main())
