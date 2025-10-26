"""
Configuration and argument parsing for qJPEG.

Handles YAML config files and command-line argument parsing.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

# Default paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BRISQUE_MODEL = str(SCRIPT_DIR / "brisque_model_live.yml")
DEFAULT_BRISQUE_RANGE = str(SCRIPT_DIR / "brisque_range_live.yml")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config file and return dict of settings.

    Automatically searches in presets/ directory if file not found directly.

    Args:
        config_path: Path to config file or preset name

    Returns:
        Dict of configuration settings
    """
    if not HAVE_YAML:
        print("[WARNING] PyYAML not installed. Config files require: pip install pyyaml")
        return {}

    p = Path(config_path)
    if not p.exists():
        # Try looking in presets/ directory
        preset_path = SCRIPT_DIR / "presets" / config_path
        if not preset_path.exists():
            preset_path = SCRIPT_DIR / "presets" / f"{config_path}.yaml"
        if preset_path.exists():
            p = preset_path
        else:
            print(f"[WARNING] Config file not found: {config_path}")
            return {}

    try:
        with open(p, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[CONFIG] Loaded: {p}")
        return config if config else {}
    except Exception as e:
        print(f"[WARNING] Failed to load config {p}: {e}")
        return {}


def save_config(config_path: str, args: argparse.Namespace):
    """
    Save current args to a YAML config file.

    Args:
        config_path: Path to save config file
        args: Parsed arguments namespace
    """
    if not HAVE_YAML:
        print("[ERROR] PyYAML not installed. Cannot save config.")
        return

    # Convert args to dict, excluding None values and input_root
    config = {}
    for key, value in vars(args).items():
        if key == 'input_root' or key == 'config' or key == 'save_config':
            continue
        if value is not None and value != argparse.SUPPRESS:
            # Skip default values for cleaner config
            if key in ['qmin', 'qmax'] and value in [1, 100]:
                continue

            # Convert comma-separated strings back to lists for proper YAML formatting
            if key == 'tiff_smart16_pct' and isinstance(value, str) and ',' in value:
                parts = [float(x.strip()) for x in value.split(',')]
                config[key] = parts
            elif key == 'auto_ev_bounds' and isinstance(value, str) and ',' in value:
                parts = [float(x.strip()) for x in value.split(',')]
                config[key] = parts
            else:
                config[key] = value

    try:
        p = Path(config_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            # Use custom representer for lists to make them inline [x, y] instead of block style
            class FlowListDumper(yaml.SafeDumper):
                pass
            def represent_list(dumper, data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            FlowListDumper.add_representer(list, represent_list)

            yaml.dump(config, f, Dumper=FlowListDumper, default_flow_style=False, sort_keys=False)
        print(f"[CONFIG] Saved to: {p}")
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")


def parse_args():
    """
    Parse command-line arguments with config file support.

    Supports two-pass parsing: loads config file first, then CLI args override.

    Returns:
        Parsed arguments namespace
    """
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
    # SSIM/search speedups
    p.add_argument("--ssim-downsample", type=int, default=4,
                   help="Compute SSIM on a 1/N grid (e.g., 4 → img[::4,::4]). 1 disables downsampling.")
    p.add_argument("--ssim-luma-only", action="store_true",
                   help="Compute SSIM on luma only (Y), faster and usually sufficient.")
    p.add_argument("--search-optimize", action="store_true",
                   help="Use Pillow optimize=True during quality search (slower). Default: off.")
    p.add_argument("--brisque-model", type=str, default=os.environ.get("BRISQUE_MODEL", DEFAULT_BRISQUE_MODEL),
                   help="Path to BRISQUE_model_live.yml")
    p.add_argument("--brisque-range", type=str, default=os.environ.get("BRISQUE_RANGE", DEFAULT_BRISQUE_RANGE),
                   help="Path to BRISQUE_range_live.yml")
    p.add_argument("--no-brisque", action="store_true", help="Disable BRISQUE scoring to speed up.")
    p.add_argument("--exiftool-mode", type=str, choices=["all","none"], default="all",
                   help="Copy metadata with exiftool after save. 'none' skips the external call (faster).")
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
    p.add_argument("--smart16-downsample", type=int, default=8,
                   help="Subsample step for percentile stretch (e.g., 8 → use every 8th pixel).")
    p.add_argument("--demosaic", type=str, default="AHD",
                   choices=["AHD", "LINEAR", "AMAZE"],
                   help="RAW demosaic algorithm (default: AHD). AMAZE needs GPL3 libraw; will fallback if unavailable.")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable progress bar / ETA output.")
    # New TIFF handling options
    p.add_argument("--tiff-apply-icc", action="store_true",
                   help="If TIFF has an embedded ICC profile, convert to sRGB using it (Pillow path only).")
    p.add_argument("--tiff-gamma", type=float, default=None,
                   help="Apply display gamma to linear TIFFs after 16→8 normalization (e.g., 2.2).")
    p.add_argument("--tiff-exposure-ev", type=float, default=0.0,
                   help="Exposure compensation in EV (applied before gamma; e.g., 1.0 doubles brightness).")
    p.add_argument("--tiff-reader", type=str, choices=["auto", "pillow", "tifffile"], default="auto",
                   help=("Which TIFF loader to use. 'auto' tries Pillow then falls back to tifffile. "
                         "'tifffile' forces percentile smart16 path; 'pillow' uses ICC and EV/Gamma only."))
    # Debug and tonemapping options
    p.add_argument("--debug", action="store_true", help="Print per-file mapping details.")
    p.add_argument("--debug-json", action="store_true", help="Emit per-file debug as JSON lines.")
    p.add_argument("--tiff-float-tonemap", type=str, choices=["none", "reinhard", "aces"],
                   default="none", help="Optional tone mapping for float TIFFs (applied after EV, before gamma).")
    # New Auto-EV options
    p.add_argument("--auto-ev-mode", type=str, choices=["off","mid","mid_guard"], default="off",
                   help="Auto exposure per image. 'mid' matches a mid-tone target; 'mid_guard' also caps highlights.")
    p.add_argument("--auto-ev-mid", type=float, default=0.18,
                   help="Target mid luminance (after tonemap, before gamma). Typical 0.16–0.22.")
    p.add_argument("--auto-ev-mid-pct", type=float, default=50.0,
                   help="Which luminance percentile to anchor for the mid (default median=50).")
    p.add_argument("--auto-ev-hi-pct", type=float, default=99.0,
                   help="Highlight percentile to protect (default 99.0).")
    p.add_argument("--auto-ev-hi-cap", type=float, default=0.90,
                   help="Max allowed highlight luminance after tonemap (default 0.90).")
    p.add_argument("--auto-ev-downsample", type=int, default=8,
                   help="Subsample step for EV solving (e.g., 8 means img[::8,::8]).")
    p.add_argument("--auto-ev-bounds", type=str, default="-4,6",
                   help="EV search bounds as 'lo,hi' (default -4..+6).")
    p.add_argument("--auto-ev-iters", type=int, default=16,
                   help="Bisection iterations (default 16).")
    # Back-compat (no longer used):
    p.add_argument("--auto-ev-percentile", type=float, default=50.0,
                   help="[DEPRECATED] Old auto-EV percentile option (ignored if --auto-ev-mode is used).")
    # Post-gamma shaping options
    p.add_argument("--blackpoint-pct", type=float, default=None,
                   help="After gamma, map this luminance percentile to 0 (e.g., 0.2).")
    p.add_argument("--whitepoint-pct", type=float, default=None,
                   help="After gamma, map this luminance percentile to 1 (e.g., 99.7).")
    p.add_argument("--shadows", type=float, default=0.0,
                   help="Shadow lift amount (0=no change, 0.1–0.3 brightens dark areas without affecting highlights).")
    p.add_argument("--contrast", type=float, default=0.0,
                   help="Post-gamma S-curve strength (0=no change, try 0.08–0.20).")
    p.add_argument("--saturation", type=float, default=1.0,
                   help="Post-gamma saturation multiplier (1=no change, e.g., 1.05).")
    # Config file support
    p.add_argument("--config", type=str, default=None,
                   help="Load settings from YAML config file. Can be a path or preset name (e.g., 'hdr-default').")
    p.add_argument("--save-config", type=str, default=None,
                   help="Save current settings to YAML config file and exit.")

    # Two-pass parsing: load config first, then override with CLI args
    # First pass: parse to check for --config
    args_temp, _ = p.parse_known_args()

    # Load config if specified
    config_dict = {}
    if args_temp.config:
        config_dict = load_config(args_temp.config)

    # Apply config as defaults
    if config_dict:
        # Convert config values to argparse defaults
        for key, value in config_dict.items():
            # Handle special cases
            if key == 'tiff_smart16_pct' and isinstance(value, list):
                config_dict[key] = f"{value[0]},{value[1]}"
            elif key == 'auto_ev_bounds' and isinstance(value, list):
                config_dict[key] = f"{value[0]},{value[1]}"

        # Set defaults from config
        for action in p._actions:
            if action.dest in config_dict and action.dest != 'config':
                # Convert underscores to match config keys
                config_key = action.dest.replace('_', '_')
                if config_key in config_dict:
                    action.default = config_dict[config_key]

    # Second pass: parse all args (CLI args override config)
    args = p.parse_args()

    # Handle save-config
    if args.save_config:
        save_config(args.save_config, args)
        import sys
        sys.exit(0)

    return args
