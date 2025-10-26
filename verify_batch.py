#!/usr/bin/env python3
"""
Batch metadata verification for qJPEG output.

Compares all files in source directory with corresponding files in
output directory and generates a comprehensive report.

Usage:
    python verify_batch.py "/path/to/Camera Roll" [--verbose] [--json]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json

try:
    from metadata_verify import verify_metadata, verify_icc_profile, CRITICAL_FIELDS, IMPORTANT_FIELDS
except ImportError:
    print("[ERROR] Cannot import metadata_verify.py - make sure it's in the same directory")
    sys.exit(1)


def find_file_pairs(input_root: Path, output_root: Path) -> List[Tuple[Path, Path]]:
    """
    Find all source/destination file pairs.

    Args:
        input_root: Source directory (e.g., "Camera Roll")
        output_root: Output directory (e.g., "Camera Roll_compressed")

    Returns:
        List of (source_path, dest_path) tuples
    """
    pairs = []

    # Supported source extensions
    src_exts = {'.tif', '.tiff', '.dng', '.cr2', '.cr3', '.nef', '.arw',
                '.orf', '.rw2', '.raf', '.srw', '.pef', '.jpg', '.jpeg'}

    # Find all source files
    for src_path in input_root.rglob('*'):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in src_exts:
            continue

        # Calculate expected destination path
        rel_path = src_path.relative_to(input_root)
        dst_path = output_root / rel_path.with_suffix('.jpg')

        if dst_path.exists():
            pairs.append((src_path, dst_path))
        else:
            print(f"[SKIP] No output for: {src_path.name}")

    return pairs


def verify_all(input_root: Path, critical_only: bool = True,
               verbose: bool = False) -> Dict:
    """
    Verify all file pairs in the directory.

    Returns:
        Summary dict with statistics and issues
    """
    output_root = input_root.parent / f"{input_root.name}_compressed"

    if not output_root.exists():
        print(f"[ERROR] Output directory not found: {output_root}")
        return {"error": "Output directory not found"}

    print(f"[SCAN] Finding file pairs...")
    print(f"  Source: {input_root}")
    print(f"  Output: {output_root}")

    pairs = find_file_pairs(input_root, output_root)

    if not pairs:
        print(f"[ERROR] No file pairs found!")
        return {"error": "No file pairs found"}

    print(f"[SCAN] Found {len(pairs)} file pairs to verify\n")

    # Statistics
    total = len(pairs)
    verified = 0
    failed = 0
    warnings = 0

    # Collect all issues
    all_issues = []
    files_with_issues = []

    # Verify each pair
    for idx, (src_path, dst_path) in enumerate(pairs, 1):
        if verbose:
            print(f"\n[{idx}/{total}] Verifying: {src_path.name}")
        else:
            # Progress indicator
            if idx % 10 == 0 or idx == total:
                print(f"  Progress: {idx}/{total} files verified...", end='\r')

        # Run verification
        issues = verify_metadata(src_path, dst_path, critical_only=critical_only, verbose=verbose)

        # Check ICC profile separately
        icc_ok = verify_icc_profile(src_path, dst_path)
        if not icc_ok:
            issues.append({
                "field": "ICC_Profile",
                "src_value": "Present",
                "dst_value": "Missing or different",
                "severity": "WARNING"
            })

        if issues:
            # Count severity
            critical_count = sum(1 for i in issues if i['severity'] == 'CRITICAL')
            warning_count = sum(1 for i in issues if i['severity'] == 'WARNING')

            if critical_count > 0:
                failed += 1
            else:
                warnings += 1

            # Store for report
            files_with_issues.append({
                "source": str(src_path),
                "destination": str(dst_path),
                "issues": issues,
                "critical_count": critical_count,
                "warning_count": warning_count,
            })

            all_issues.extend(issues)
        else:
            verified += 1

    if not verbose:
        print()  # Clear progress line

    # Generate summary
    summary = {
        "total": total,
        "verified": verified,
        "failed": failed,
        "warnings": warnings,
        "files_with_issues": len(files_with_issues),
        "issue_details": files_with_issues,
    }

    # Count most common issues
    field_counts = {}
    for issue in all_issues:
        field = issue['field']
        field_counts[field] = field_counts.get(field, 0) + 1

    summary["common_issues"] = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)

    return summary


def print_summary(summary: Dict, verbose: bool = False):
    """Print human-readable summary report."""
    print("\n" + "="*70)
    print("METADATA VERIFICATION REPORT")
    print("="*70)

    if "error" in summary:
        print(f"\n[ERROR] {summary['error']}")
        return

    # Statistics
    print(f"\nStatistics:")
    print(f"  Total files:       {summary['total']}")
    print(f"  ✓ Perfect match:   {summary['verified']}")
    print(f"  ⚠ Warnings:        {summary['warnings']}")
    print(f"  ✗ Critical issues: {summary['failed']}")

    # Success rate
    success_rate = (summary['verified'] / summary['total'] * 100) if summary['total'] > 0 else 0
    print(f"\n  Success rate:      {success_rate:.1f}%")

    # Common issues
    if summary['common_issues']:
        print(f"\nMost Common Issues:")
        for field, count in summary['common_issues'][:10]:
            print(f"  {field:30s} {count:3d} files")

    # Files with critical issues
    critical_files = [f for f in summary['issue_details'] if f['critical_count'] > 0]
    if critical_files:
        print(f"\n⚠ Files with CRITICAL metadata issues: {len(critical_files)}")
        for file_info in critical_files[:5]:  # Show first 5
            src = Path(file_info['source'])
            print(f"\n  {src.name}:")
            for issue in file_info['issues']:
                if issue['severity'] == 'CRITICAL':
                    print(f"    [CRITICAL] {issue['field']}: "
                          f"{issue['src_value']} → {issue['dst_value']}")

        if len(critical_files) > 5:
            print(f"\n  ... and {len(critical_files) - 5} more files with critical issues")

    # Files with warnings
    warning_files = [f for f in summary['issue_details'] if f['critical_count'] == 0 and f['warning_count'] > 0]
    if warning_files and verbose:
        print(f"\nℹ Files with warnings: {len(warning_files)}")
        for file_info in warning_files[:3]:  # Show first 3
            src = Path(file_info['source'])
            print(f"\n  {src.name}:")
            for issue in file_info['issues']:
                print(f"    [WARNING] {issue['field']}: "
                      f"{issue['src_value']} → {issue['dst_value']}")

        if len(warning_files) > 3:
            print(f"\n  ... and {len(warning_files) - 3} more files with warnings")

    # Overall verdict
    print("\n" + "="*70)
    if summary['failed'] == 0:
        if summary['warnings'] == 0:
            print("✓ ALL FILES VERIFIED SUCCESSFULLY")
        else:
            print(f"✓ No critical issues, but {summary['warnings']} files have warnings")
    else:
        print(f"✗ {summary['failed']} files have CRITICAL metadata issues!")
        print("  Review the issues above and check metadata preservation settings.")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch metadata verification for qJPEG output"
    )
    parser.add_argument("input_root", type=str,
                       help="Source directory (e.g., '/path/to/Camera Roll')")
    parser.add_argument("--critical-only", action="store_true",
                       help="Only check critical fields (faster)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output for each file")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON report instead of text")

    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()

    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}")
        return 1

    # Run verification
    summary = verify_all(input_root, critical_only=args.critical_only, verbose=args.verbose)

    # Output results
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary, verbose=args.verbose)

    # Exit code
    if "error" in summary:
        return 1
    elif summary['failed'] > 0:
        return 2  # Critical issues found
    elif summary['warnings'] > 0:
        return 0  # Warnings but no critical issues
    else:
        return 0  # All good


if __name__ == "__main__":
    sys.exit(main())
