"""
Metadata handling for qJPEG.

Functions for copying EXIF, IPTC, XMP metadata and sidecar files.
"""

import shutil
import subprocess
from pathlib import Path

from .utils import ensure_dir, SIDECAR_EXTS, has_exiftool

# Check if exiftool is available
EXIFTOOL_OK = has_exiftool()


def copy_sidecars(src_path: Path, dst_path_without_ext: Path):
    """
    Copy sidecar files (e.g., .xmp, .xml, .json) to the mirrored output path.

    Args:
        src_path: Source image file path
        dst_path_without_ext: Destination path without extension (will add sidecar extensions)
    """
    stem = src_path.with_suffix("")
    for ext in SIDECAR_EXTS:
        sidecar = stem.with_suffix(ext)
        if sidecar.exists():
            dst_sidecar = dst_path_without_ext.with_suffix(ext)
            ensure_dir(dst_sidecar)
            shutil.copy2(sidecar, dst_sidecar)


def copy_all_metadata_with_exiftool(src_path: Path, dst_path: Path):
    """
    Copy ALL tags (EXIF/IPTC/XMP) from src to dst using exiftool, if available.
    Also sets filesystem modification/creation dates to match EXIF DateTimeOriginal.

    Args:
        src_path: Source image file
        dst_path: Destination JPEG file
    """
    if not EXIFTOOL_OK:
        return
    cmd = [
        "exiftool",
        "-overwrite_original",
        "-All:All",
        "-TagsFromFile", str(src_path),
        # Set filesystem dates to match EXIF DateTimeOriginal
        "-FileModifyDate<DateTimeOriginal",
        "-FileCreateDate<DateTimeOriginal",  # Works on macOS/Windows, ignored on Linux ext4
        str(dst_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
