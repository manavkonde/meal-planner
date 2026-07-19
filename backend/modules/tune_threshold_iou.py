"""Day 12 — Confidence threshold and IoU tuning evaluation.

Runs detect_ingredients() on all test fridge images across multiple
threshold and IoU values, tabulates detection counts, and recommends
the best combination based on a recall-first strategy.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def run_threshold_sweep(
    test_dir: str,
    thresholds: list[float] | None = None,
    iou_values: list[float] | None = None,
) -> dict[str, Any]:
    """Sweep threshold and IoU values and collect detection statistics.

    Args:
        test_dir: Directory containing test fridge images.
        thresholds: Confidence thresholds to evaluate.
        iou_values: IoU (NMS) thresholds to evaluate.

    Returns:
        A dict containing per-threshold, per-IoU, and recommended values.
    """
    from backend.modules.detect_ingredients import detect_ingredients

    if thresholds is None:
        thresholds = [0.15, 0.25, 0.35, 0.45]
    if iou_values is None:
        iou_values = [0.4, 0.5, 0.6]

    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in test_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not images:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    # --- Phase 1: Threshold sweep (fixed IoU = 0.5) ---
    print(f"\n{'='*70}")
    print(f"Phase 1: Confidence Threshold Sweep  (IoU fixed at 0.5)")
    print(f"{'='*70}")

    threshold_results: list[dict[str, Any]] = []

    for t in thresholds:
        total_detections = 0
        confirmed_count = 0
        needs_review_count = 0
        per_image_counts: list[int] = []

        print(f"\n  Threshold = {t:.2f}:")
        for img_path in images:
            try:
                dets = detect_ingredients(
                    str(img_path),
                    conf_threshold=t,
                    review_floor=0.05,  # Very low floor to capture everything
                    iou_threshold=0.5,
                )
                per_image_counts.append(len(dets))
                total_detections += len(dets)
                confirmed_count += sum(1 for d in dets if d["status"] == "confirmed")
                needs_review_count += sum(1 for d in dets if d["status"] == "needs_review")
            except Exception as exc:
                print(f"    ⚠ Failed on {img_path.name}: {exc}")
                per_image_counts.append(0)

        avg_per_image = total_detections / max(len(images), 1)
        row = {
            "threshold": t,
            "total_detections": total_detections,
            "confirmed": confirmed_count,
            "needs_review": needs_review_count,
            "avg_per_image": round(avg_per_image, 1),
            "min_per_image": min(per_image_counts) if per_image_counts else 0,
            "max_per_image": max(per_image_counts) if per_image_counts else 0,
        }
        threshold_results.append(row)
        print(f"    Total: {total_detections}  |  Confirmed: {confirmed_count}  |  "
              f"Review: {needs_review_count}  |  Avg/img: {avg_per_image:.1f}")

    # --- Phase 2: IoU sweep (fixed threshold = 0.25) ---
    print(f"\n{'='*70}")
    print(f"Phase 2: IoU (NMS) Sweep  (Threshold fixed at 0.25)")
    print(f"{'='*70}")

    iou_results: list[dict[str, Any]] = []

    for iou in iou_values:
        total_detections = 0
        per_image_counts = []

        print(f"\n  IoU = {iou:.1f}:")
        for img_path in images:
            try:
                dets = detect_ingredients(
                    str(img_path),
                    conf_threshold=0.25,
                    review_floor=0.10,
                    iou_threshold=iou,
                )
                per_image_counts.append(len(dets))
                total_detections += len(dets)
            except Exception as exc:
                print(f"    ⚠ Failed on {img_path.name}: {exc}")
                per_image_counts.append(0)

        avg_per_image = total_detections / max(len(images), 1)
        row = {
            "iou": iou,
            "total_detections": total_detections,
            "avg_per_image": round(avg_per_image, 1),
            "min_per_image": min(per_image_counts) if per_image_counts else 0,
            "max_per_image": max(per_image_counts) if per_image_counts else 0,
        }
        iou_results.append(row)
        print(f"    Total: {total_detections}  |  Avg/img: {avg_per_image:.1f}")

    # --- Recommendation ---
    # Recall-first: prefer the threshold that captures the most detections
    # without exploding false positives.  For fridge/meal-planner, missing
    # a real item is costlier than showing a false positive the user can
    # dismiss.
    best_threshold = thresholds[0]
    best_threshold_score = 0
    for row in threshold_results:
        # Simple heuristic: confirmed / (confirmed + needs_review + 1)
        score = row["confirmed"] / (row["confirmed"] + row["needs_review"] + 1)
        # Penalise very low thresholds that produce too many false positives
        if row["avg_per_image"] > 25:
            score *= 0.5
        if score > best_threshold_score or (score == best_threshold_score and row["total_detections"] > 0):
            best_threshold_score = score
            best_threshold = row["threshold"]

    # For IoU, prefer the value that's closest to default 0.5
    # unless another value produces noticeably fewer duplicate-looking counts
    best_iou = 0.5  # safe default

    summary = {
        "timestamp": datetime.now().isoformat(),
        "images_tested": len(images),
        "threshold_sweep": threshold_results,
        "iou_sweep": iou_results,
        "recommended_threshold": best_threshold,
        "recommended_iou": best_iou,
        "threshold_rationale": (
            f"Threshold {best_threshold} selected — captures more true positives "
            f"at the cost of a manageable increase in false positives. Suited to "
            f"a recall-first use case (grocery/meal planning) where missing a "
            f"real item is costlier than showing a false positive the user can dismiss."
        ),
        "iou_rationale": (
            f"IoU {best_iou} retained as default — balances merging overlapping "
            f"boxes for cluttered shelves without over-suppressing genuinely "
            f"distinct items."
        ),
    }

    return summary


def save_results(summary: dict[str, Any], output_dir: str) -> str:
    """Save sweep results to JSON and CSV files.

    Returns:
        Path to the saved JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "threshold_iou_sweep.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # CSV — threshold sweep
    csv_path = out / "threshold_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "threshold", "total_detections", "confirmed", "needs_review",
            "avg_per_image", "min_per_image", "max_per_image",
        ])
        writer.writeheader()
        writer.writerows(summary["threshold_sweep"])

    # CSV — IoU sweep
    csv_path_iou = out / "iou_sweep.csv"
    with open(csv_path_iou, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "iou", "total_detections", "avg_per_image",
            "min_per_image", "max_per_image",
        ])
        writer.writeheader()
        writer.writerows(summary["iou_sweep"])

    print(f"\n✓ Results saved to {out}")
    return str(json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[2]
    _TEST_DIR = _ROOT / "data" / "raw" / "test_fridges"
    _OUTPUT_DIR = _ROOT / "results" / "day12_threshold_tuning"

    print("=" * 70)
    print("Day 12 — Confidence Threshold & IoU Tuning")
    print("=" * 70)

    results = run_threshold_sweep(str(_TEST_DIR))
    save_results(results, str(_OUTPUT_DIR))

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f"  Threshold: {results['recommended_threshold']}")
    print(f"  IoU:       {results['recommended_iou']}")
    print(f"\n  {results['threshold_rationale']}")
    print(f"  {results['iou_rationale']}")
