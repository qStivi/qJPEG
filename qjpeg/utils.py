"""
Utility functions for qJPEG.

File system operations, path manipulation, and helper functions.
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Set, List, Dict

# File extension constants
IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}
RAW_EXTS: Set[str] = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".srw", ".pef"}
SIDECAR_EXTS: Set[str] = {".xmp", ".xml", ".json"}


def ensure_dir(path: Path):
    """Ensure parent directory exists for given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def has_exiftool() -> bool:
    """Check if exiftool is available on the system."""
    try:
        subprocess.run(["exiftool", "-ver"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def short_hash(text: str, n=4) -> str:
    """Generate short hash for deduplication."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def dest_path_for(
        src_path: Path,
        input_root: Path,
        out_root: Path,
        mirror_structure: bool,
        flat_dedupe: bool,
) -> Tuple[Optional[Path], bool]:
    """
    Compute destination path (.jpg) and a flag whether it collides in flat mode.

    Returns:
        (destination_path, is_collision)
    """
    ext = src_path.suffix.lower()
    rel = src_path.relative_to(input_root) if mirror_structure else Path(src_path.name)
    dst_rel = rel.with_suffix(".jpg") if (ext in RAW_EXTS or ext in IMG_EXTS) else None
    if dst_rel is None:
        return None, False
    if mirror_structure:
        return out_root / dst_rel, False
    # flat mode: dedupe by adding a short hash from the relative path (without extension)
    base = dst_rel.stem
    hashed = f"{base}__{short_hash(str(rel.with_suffix('')))}.jpg" if flat_dedupe else f"{base}.jpg"
    return out_root / hashed, (not flat_dedupe)


def collect_sources(input_root: Path, allow_exts: Optional[Set[str]]) -> List[Path]:
    """
    Collect all source image files from input directory.

    Args:
        input_root: Root directory to search
        allow_exts: Set of allowed extensions (e.g., {'.tif', '.dng'}), None for all

    Returns:
        List of source file paths
    """
    files: List[Path] = []
    for p in input_root.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if allow_exts is not None and ext not in allow_exts:
            continue
        if ext in RAW_EXTS or ext in IMG_EXTS:
            files.append(p)
    return files


def find_flat_collisions(files: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Find filename collisions when using flat output mode.

    Args:
        files: List of source file paths

    Returns:
        List of (first_file, duplicate_file) tuples
    """
    seen: Dict[str, Path] = {}
    dups: List[Tuple[Path, Path]] = []
    for p in files:
        name = p.name
        if name in seen:
            dups.append((seen[name], p))
        else:
            seen[name] = p
    return dups
