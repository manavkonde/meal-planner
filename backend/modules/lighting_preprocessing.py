"""Day 12 — Lighting preprocessing for fridge/pantry images.

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to
compensate for dim or uneven lighting common in real fridge photos.
The enhancement is applied only to the luminance channel in LAB colour
space so that colour information is preserved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def enhance_lighting(
    image_path: str,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE lighting enhancement to a single image.

    Args:
        image_path: Absolute or relative path to the input image.
        clip_limit: Contrast-limiting threshold for CLAHE.
        tile_grid_size: Size of the grid for histogram equalisation.

    Returns:
        Enhanced BGR image as a NumPy array.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        ValueError: If the image cannot be decoded by OpenCV.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to decode image: {image_path}")

    return enhance_lighting_array(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


def enhance_lighting_array(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE lighting enhancement to a BGR NumPy array.

    This variant is useful when the image has already been loaded into
    memory (e.g. by the detection pipeline).

    Args:
        img: BGR image array (H×W×3, dtype ``uint8``).
        clip_limit: Contrast-limiting threshold for CLAHE.
        tile_grid_size: Size of the grid for histogram equalisation.

    Returns:
        Enhanced BGR image as a NumPy array with same shape and dtype.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# A/B comparison helper — runs detection with and without enhancement
# ---------------------------------------------------------------------------

def test_lighting_improvement(
    test_dir: str,
    detect_fn: Any = None,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compare detection results with and without lighting enhancement.

    For each image in *test_dir*, runs the detection function twice (on
    the original image and on an enhanced copy saved as a temp file) and
    records the delta in detection count.

    Args:
        test_dir: Directory containing test fridge images.
        detect_fn: A callable ``(image_path, conf_threshold) → list[dict]``.
                   If ``None``, a lazy import of ``detect_ingredients`` is used.
        conf_threshold: Confidence threshold forwarded to the detector.

    Returns:
        A summary dict with per-image and aggregate results.
    """
    if detect_fn is None:
        from backend.modules.detect_ingredients import detect_ingredients
        detect_fn = detect_ingredients

    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in test_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    per_image: list[dict[str, Any]] = []
    total_original = 0
    total_enhanced = 0

    for img_path in images:
        # Original detections
        try:
            original_dets = detect_fn(str(img_path), conf_threshold=conf_threshold)
        except Exception as exc:
            original_dets = []
            print(f"  ⚠ Detection failed on original {img_path.name}: {exc}")

        # Enhanced detections — save enhanced image to a temp file
        try:
            enhanced_img = enhance_lighting(str(img_path))
            temp_path = img_path.parent / f"_enhanced_tmp_{img_path.name}"
            cv2.imwrite(str(temp_path), enhanced_img)
            enhanced_dets = detect_fn(str(temp_path), conf_threshold=conf_threshold)
            os.remove(temp_path)
        except Exception as exc:
            enhanced_dets = []
            print(f"  ⚠ Detection failed on enhanced {img_path.name}: {exc}")

        delta = len(enhanced_dets) - len(original_dets)
        per_image.append({
            "image": img_path.name,
            "original_count": len(original_dets),
            "enhanced_count": len(enhanced_dets),
            "delta": delta,
        })
        total_original += len(original_dets)
        total_enhanced += len(enhanced_dets)

    improvement = total_enhanced - total_original
    improvement_pct = (improvement / max(total_original, 1)) * 100

    summary = {
        "images_tested": len(images),
        "total_original_detections": total_original,
        "total_enhanced_detections": total_enhanced,
        "net_detection_change": improvement,
        "net_detection_change_pct": round(improvement_pct, 1),
        "recommendation": (
            "ADOPT — enhancement meaningfully increases detections"
            if improvement_pct > 5
            else "SKIP — enhancement does not meaningfully improve detection"
        ),
        "per_image": per_image,
    }

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    _ROOT = Path(__file__).resolve().parents[2]
    _TEST_DIR = _ROOT / "data" / "raw" / "test_fridges"

    print("=" * 70)
    print("Day 12 — Lighting Enhancement A/B Test")
    print("=" * 70)

    result = test_lighting_improvement(str(_TEST_DIR))

    print(f"\nImages tested: {result['images_tested']}")
    print(f"Total original detections:  {result['total_original_detections']}")
    print(f"Total enhanced detections:  {result['total_enhanced_detections']}")
    print(f"Net change: {result['net_detection_change']:+d} ({result['net_detection_change_pct']:+.1f}%)")
    print(f"\nRecommendation: {result['recommendation']}")

    # Save detailed results
    out_dir = _ROOT / "results" / "day12_lighting"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lighting_ab_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")
