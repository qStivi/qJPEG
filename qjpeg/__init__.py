"""
qJPEG - Intelligent JPEG optimizer with HDR/RAW support.

Batch JPEG optimizer with SSIM-guided quality search, robust RAW/TIFF loading,
metadata preservation, and optional BRISQUE scoring.
"""

__version__ = "2.0.0"

# Core functionality
from .pipeline import process_tree, process_one, _play_finish_sound
from .image_io import load_image_as_rgb
from .image_processing import MapStats
from .quality import ssim_threshold_search, brisque_score_cv2, save_final_jpeg
from .metadata import copy_sidecars, copy_all_metadata_with_exiftool
from .utils import (
    ensure_dir,
    has_exiftool,
    dest_path_for,
    collect_sources,
    find_flat_collisions,
    IMG_EXTS,
    RAW_EXTS,
    SIDECAR_EXTS,
)

__all__ = [
    # Version
    "__version__",
    # Pipeline
    "process_tree",
    "process_one",
    "_play_finish_sound",
    # Image I/O
    "load_image_as_rgb",
    # Image processing
    "MapStats",
    # Quality
    "ssim_threshold_search",
    "brisque_score_cv2",
    "save_final_jpeg",
    # Metadata
    "copy_sidecars",
    "copy_all_metadata_with_exiftool",
    # Utils
    "ensure_dir",
    "has_exiftool",
    "dest_path_for",
    "collect_sources",
    "find_flat_collisions",
    "IMG_EXTS",
    "RAW_EXTS",
    "SIDECAR_EXTS",
]
