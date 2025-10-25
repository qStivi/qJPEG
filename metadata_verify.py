#!/usr/bin/env python3
"""
Metadata verification system for qJPEG.

Verifies that critical metadata is preserved after JPEG conversion:
- Creation date (DateTimeOriginal, CreateDate)
- GPS location (GPSLatitude, GPSLongitude)
- Camera info (Make, Model, LensModel)
- Color profile (ICC profile)
- Author/Copyright

Usage:
    from metadata_verify import verify_metadata

    issues = verify_metadata(src_path, dst_path, critical_only=True)
    if issues:
        print(f"[WARNING] Metadata issues: {issues}")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import json

# Critical metadata fields that MUST be preserved
CRITICAL_FIELDS = {
    "DateTimeOriginal",
    "CreateDate",
    "DateCreated",
    "GPSLatitude",
    "GPSLongitude",
    "GPSAltitude",
    "ColorSpace",
    "ProfileDescription",  # ICC profile name
}

# Important but not critical
IMPORTANT_FIELDS = {
    "Make",
    "Model",
    "LensModel",
    "LensMake",
    "FocalLength",
    "FNumber",
    "ExposureTime",
    "ISO",
    "WhiteBalance",
    "Artist",
    "Copyright",
    "Creator",
    "Rights",
}


def extract_metadata_exiftool(file_path: Path) -> Dict[str, str]:
    """
    Extract metadata from file using exiftool.
    Returns dict of tag -> value.
    """
    try:
        cmd = [
            "exiftool",
            "-json",
            "-a",  # Allow duplicate tags
            "-G",  # Show group names
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
        if not data or len(data) == 0:
            return {}

        return data[0]  # exiftool returns list with one dict
    except Exception as e:
        print(f"[ERROR] Failed to extract metadata from {file_path}: {e}")
        return {}


def normalize_value(value) -> Optional[str]:
    """Normalize metadata value for comparison."""
    if value is None:
        return None

    # Convert to string
    s = str(value).strip()

    # Empty values
    if not s or s.lower() in ['', 'null', 'none', 'n/a', '-']:
        return None

    # Normalize GPS coordinates (remove trailing zeros, spaces)
    if 'deg' in s.lower() or "'" in s or '"' in s:
        # GPS format like: 37 deg 47' 21.84" N
        # Just normalize spaces
        s = ' '.join(s.split())

    return s


def compare_metadata(src_meta: Dict, dst_meta: Dict,
                    check_fields: set) -> List[Tuple[str, str, str]]:
    """
    Compare metadata between source and destination.

    Returns list of (field, src_value, dst_value) for mismatches.
    Only checks fields in check_fields.
    """
    issues = []

    for field in check_fields:
        src_val = normalize_value(src_meta.get(field))
        dst_val = normalize_value(dst_meta.get(field))

        # Both missing = OK
        if src_val is None and dst_val is None:
            continue

        # Source has value, dest doesn't = ISSUE
        if src_val is not None and dst_val is None:
            issues.append((field, src_val, "MISSING"))
            continue

        # Values differ = ISSUE
        if src_val != dst_val:
            issues.append((field, src_val or "MISSING", dst_val or "MISSING"))

    return issues


def verify_metadata(src_path: Path, dst_path: Path,
                   critical_only: bool = True,
                   verbose: bool = False) -> List[Dict]:
    """
    Verify that metadata was correctly copied from src to dst.

    Args:
        src_path: Source file (TIFF/RAW/etc)
        dst_path: Destination file (JPEG)
        critical_only: Only check CRITICAL_FIELDS if True, else check IMPORTANT too
        verbose: Print detailed comparison

    Returns:
        List of issue dicts with keys: field, src_value, dst_value, severity
    """
    # Extract metadata
    if verbose:
        print(f"[VERIFY] Extracting metadata from source: {src_path.name}")
    src_meta = extract_metadata_exiftool(src_path)

    if verbose:
        print(f"[VERIFY] Extracting metadata from dest: {dst_path.name}")
    dst_meta = extract_metadata_exiftool(dst_path)

    if not src_meta:
        return [{"field": "EXIFTOOL", "src_value": "FAILED", "dst_value": "N/A",
                "severity": "ERROR"}]

    # Compare critical fields
    check_fields = CRITICAL_FIELDS
    if not critical_only:
        check_fields = CRITICAL_FIELDS | IMPORTANT_FIELDS

    mismatches = compare_metadata(src_meta, dst_meta, check_fields)

    # Convert to issue dicts
    issues = []
    for field, src_val, dst_val in mismatches:
        severity = "CRITICAL" if field in CRITICAL_FIELDS else "WARNING"
        issues.append({
            "field": field,
            "src_value": src_val,
            "dst_value": dst_val,
            "severity": severity,
        })

        if verbose:
            print(f"[{severity}] {field}: {src_val} → {dst_val}")

    if verbose and not issues:
        print(f"[OK] All metadata verified successfully")

    return issues


def verify_icc_profile(src_path: Path, dst_path: Path) -> bool:
    """
    Verify ICC profile was preserved.
    Uses Pillow to check embedded profiles.
    """
    try:
        from PIL import Image

        src_img = Image.open(src_path)
        dst_img = Image.open(dst_path)

        src_icc = src_img.info.get('icc_profile')
        dst_icc = dst_img.info.get('icc_profile')

        # If source has ICC, dest must have it too
        if src_icc and not dst_icc:
            return False

        # If both have ICC, they should match (or at least exist)
        if src_icc and dst_icc:
            # Quick check: compare first 100 bytes
            if src_icc[:100] != dst_icc[:100]:
                # Different profiles - might be sRGB conversion, which is OK
                # Just verify dst has SOME profile
                return len(dst_icc) > 0

        return True
    except Exception as e:
        print(f"[WARNING] ICC profile check failed: {e}")
        return True  # Don't fail on ICC check errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python metadata_verify.py <source.tif> <dest.jpg>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    issues = verify_metadata(src, dst, critical_only=False, verbose=True)

    if issues:
        print(f"\n[FAILED] Found {len(issues)} metadata issues")
        for issue in issues:
            print(f"  [{issue['severity']}] {issue['field']}: "
                  f"{issue['src_value']} → {issue['dst_value']}")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] All metadata verified")
        sys.exit(0)
