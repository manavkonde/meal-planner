"""Day 12 — Evaluate updated detect_ingredients() on test fridge images.

Produces a before/after/after-edge-case comparison table and summary
statistics showing the progression from COCO baseline → fine-tuned model
→ fine-tuned model with edge-case handling (Day 12 improvements).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def evaluate_detection(
    test_dir: str,
    conf_threshold: float = 0.25,
    review_floor: float = 0.10,
    iou_threshold: float = 0.5,
    use_lighting: bool = False,
) -> dict[str, Any]:
    """Run the updated detect_ingredients on all test images.

    Args:
        test_dir: Directory containing test fridge images.
        conf_threshold: Confidence threshold for confirmed detections.
        review_floor: Noise floor for needs_review detections.
        iou_threshold: IoU threshold for NMS.
        use_lighting: Whether to enable CLAHE lighting enhancement.

    Returns:
        Summary dict with per-image results and aggregate statistics.
    """
    from backend.modules.detect_ingredients import detect_ingredients, get_model_info

    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in test_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    model_info = get_model_info()
    per_image: list[dict[str, Any]] = []

    total_confirmed = 0
    total_review = 0
    all_classes: dict[str, int] = {}

    print(f"\nModel: {model_info.get('source', 'unknown')}")
    print(f"Settings: conf={conf_threshold}, floor={review_floor}, "
          f"iou={iou_threshold}, lighting={use_lighting}")
    print(f"Images: {len(images)}\n")

    for idx, img_path in enumerate(images):
        try:
            dets = detect_ingredients(
                str(img_path),
                conf_threshold=conf_threshold,
                review_floor=review_floor,
                iou_threshold=iou_threshold,
                use_lighting_enhancement=use_lighting,
            )
        except Exception as exc:
            print(f"  [{idx+1}/{len(images)}] ⚠ {img_path.name}: {exc}")
            per_image.append({
                "image": img_path.name,
                "total": 0,
                "confirmed": 0,
                "needs_review": 0,
                "classes": [],
                "error": str(exc),
            })
            continue

        confirmed = [d for d in dets if d["status"] == "confirmed"]
        review = [d for d in dets if d["status"] == "needs_review"]
        classes_found = [d["item"] for d in dets]

        for cls in classes_found:
            all_classes[cls] = all_classes.get(cls, 0) + 1

        total_confirmed += len(confirmed)
        total_review += len(review)

        per_image.append({
            "image": img_path.name,
            "total": len(dets),
            "confirmed": len(confirmed),
            "needs_review": len(review),
            "classes": classes_found,
        })

        status_str = f"✓{len(confirmed)} ?{len(review)}"
        print(f"  [{idx+1}/{len(images)}] {img_path.name}: {len(dets)} detections ({status_str})")

    total = total_confirmed + total_review
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "settings": {
            "conf_threshold": conf_threshold,
            "review_floor": review_floor,
            "iou_threshold": iou_threshold,
            "use_lighting": use_lighting,
        },
        "aggregate": {
            "images_tested": len(images),
            "total_detections": total,
            "total_confirmed": total_confirmed,
            "total_needs_review": total_review,
            "avg_detections_per_image": round(total / max(len(images), 1), 1),
            "avg_confirmed_per_image": round(total_confirmed / max(len(images), 1), 1),
            "unique_classes_detected": len(all_classes),
            "class_frequency": dict(sorted(all_classes.items(), key=lambda x: -x[1])),
        },
        "per_image": per_image,
    }

    return summary


def build_comparison_table(
    day12_summary: dict[str, Any],
    day5_baseline_path: str | None = None,
) -> list[dict[str, str]]:
    """Build a before/after comparison table.

    Compares Day 5 COCO baseline → Day 12 with edge-case handling.
    Day 11 (fine-tuned, no edge-case handling) would be a middle column
    when a fine-tuned checkpoint is available.

    Returns:
        A list of row dicts suitable for CSV/markdown rendering.
    """
    agg = day12_summary["aggregate"]

    # Day 5 baseline numbers (from day5_baseline_summary.md)
    day5 = {
        "total_detections": 205,
        "avg_per_image": 6.4,
        "correct_rate": "~19.5%",
        "unique_classes": 5,
        "model": "COCO pretrained (yolov8n)",
    }

    rows = [
        {
            "metric": "Model",
            "day5_baseline": day5["model"],
            "day12_edge_case": day12_summary["model_info"].get("source", "unknown"),
        },
        {
            "metric": "Total detections",
            "day5_baseline": str(day5["total_detections"]),
            "day12_edge_case": str(agg["total_detections"]),
        },
        {
            "metric": "Avg detections/image",
            "day5_baseline": str(day5["avg_per_image"]),
            "day12_edge_case": str(agg["avg_detections_per_image"]),
        },
        {
            "metric": "Unique classes detected",
            "day5_baseline": str(day5["unique_classes"]),
            "day12_edge_case": str(agg["unique_classes_detected"]),
        },
        {
            "metric": "Confirmed detections",
            "day5_baseline": "N/A (no tiering)",
            "day12_edge_case": str(agg["total_confirmed"]),
        },
        {
            "metric": "Needs-review detections",
            "day5_baseline": "N/A (no tiering)",
            "day12_edge_case": str(agg["total_needs_review"]),
        },
    ]

    return rows


def save_evaluation(
    summary: dict[str, Any],
    comparison: list[dict[str, str]],
    output_dir: str,
) -> str:
    """Save evaluation results to JSON and CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full JSON
    json_path = out / "day12_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-image CSV
    csv_path = out / "day12_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "total", "confirmed", "needs_review",
        ])
        writer.writeheader()
        for row in summary["per_image"]:
            writer.writerow({
                "image": row["image"],
                "total": row["total"],
                "confirmed": row["confirmed"],
                "needs_review": row["needs_review"],
            })

    # Comparison CSV
    cmp_path = out / "day12_comparison.csv"
    with open(cmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "day5_baseline", "day12_edge_case",
        ])
        writer.writeheader()
        writer.writerows(comparison)

    print(f"\n✓ Evaluation saved to {out}")
    return str(json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[2]
    _TEST_DIR = _ROOT / "data" / "raw" / "test_fridges"
    _OUTPUT_DIR = _ROOT / "results" / "day12_evaluation"

    print("=" * 70)
    print("Day 12 — Edge-Case Handling Evaluation")
    print("=" * 70)

    result = evaluate_detection(str(_TEST_DIR))
    comparison = build_comparison_table(result)

    save_evaluation(result, comparison, str(_OUTPUT_DIR))

    # Print comparison table
    print(f"\n{'='*70}")
    print("Before / After Comparison")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Day 5 Baseline':<25} {'Day 12 Edge-Case':<25}")
    print("-" * 80)
    for row in comparison:
        print(f"{row['metric']:<30} {row['day5_baseline']:<25} {row['day12_edge_case']:<25}")

    agg = result["aggregate"]
    print(f"\nClass frequency (top 10):")
    for cls, count in list(agg["class_frequency"].items())[:10]:
        print(f"  {cls}: {count}")
