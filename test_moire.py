#!/usr/bin/env python3
"""
Quick test script for moiré detection on screen photographs.
"""
import numpy as np
from PIL import Image
from qjpeg.quality import detect_moire_fft

def test_image(image_path):
    """Test moiré detection on a single image."""
    print(f"\n{'='*70}")
    print(f"Testing: {image_path}")
    print('='*70)

    # Load image
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)

    print(f"Image size: {img.size[0]}x{img.size[1]}")

    # Test with default thresholds
    has_moire, confidence, debug = detect_moire_fft(arr, threshold=0.10, min_peak_ratio=1.5)

    print(f"\n🔍 Moiré Detection Results:")
    print(f"   Detected: {'YES ⚠️' if has_moire else 'NO ✓'}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"   Peak count: {debug.get('peak_count', 0)}")
    print(f"   Max peak strength: {debug.get('max_peak_strength', 0):.2f}x median")
    print(f"   Median magnitude: {debug.get('median_magnitude', 0):.2f}")

    if has_moire:
        print(f"\n⚠️  This image has moiré patterns - BRISQUE scores may be unreliable!")
    else:
        print(f"\n✓  No moiré detected - BRISQUE scores should be reliable")

    return has_moire, confidence

if __name__ == "__main__":
    import sys

    # Test specific screen photos
    screen_photos = [
        r"C:\Users\steph\Pictures\Camera Roll\2024\08\IMG_20240829_130545.jpg",
    ]

    # Add additional photos if provided as arguments
    if len(sys.argv) > 1:
        screen_photos = sys.argv[1:]

    results = []
    for photo in screen_photos:
        try:
            has_moire, conf = test_image(photo)
            results.append((photo, has_moire, conf))
        except Exception as e:
            print(f"❌ Error processing {photo}: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print('='*70)
    detected = sum(1 for _, has_moire, _ in results if has_moire)
    print(f"Total images: {len(results)}")
    print(f"Moiré detected: {detected}")
    print(f"Clean images: {len(results) - detected}")
