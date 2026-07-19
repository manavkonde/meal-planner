"""Day 13 — Evaluate CLIP fallback effectiveness.

Compares YOLOv8-only detection vs. YOLOv8+CLIP hybrid detection on the
real fridge test set, focusing specifically on weak classes.  Reports
per-class improvement and latency tradeoffs.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def evaluate_clip_fallback(
    test_dir: str,
    conf_threshold: float = 0.25,
    review_floor: float = 0.10,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run YOLOv8-only vs. YOLOv8+CLIP on all test images.

    Args:
        test_dir: Directory containing test fridge images.
        conf_threshold: Confidence threshold.
        review_floor: Noise floor.
        iou_threshold: NMS IoU threshold.

    Returns:
        Summary dict with per-image comparisons and latency data.
    """
    from backend.modules.detect_ingredients import detect_ingredients
    from backend.modules.clip_fallback import (
        detect_ingredients_with_fallback,
        identify_weak_classes,
    )

    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in test_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    weak_classes = identify_weak_classes()
    per_image: list[dict[str, Any]] = []
    yolo_latencies: list[float] = []
    hybrid_latencies: list[float] = []

    # Class-level tracking
    yolo_class_counts: dict[str, int] = {}
    hybrid_class_counts: dict[str, int] = {}
    clip_relabelled: dict[str, int] = {}

    print(f"\nWeak classes ({len(weak_classes)}): {', '.join(weak_classes[:10])}...")
    print(f"Images: {len(images)}\n")

    for idx, img_path in enumerate(images):
        print(f"  [{idx+1}/{len(images)}] {img_path.name}...", end=" ", flush=True)

        # --- YOLOv8-only ---
        t0 = time.perf_counter()
        try:
            yolo_dets = detect_ingredients(
                str(img_path),
                conf_threshold=conf_threshold,
                review_floor=review_floor,
                iou_threshold=iou_threshold,
            )
        except Exception as exc:
            print(f"YOLO failed: {exc}")
            yolo_dets = []
        yolo_time = time.perf_counter() - t0
        yolo_latencies.append(yolo_time)

        # --- YOLOv8+CLIP hybrid ---
        t0 = time.perf_counter()
        try:
            hybrid_dets = detect_ingredients_with_fallback(
                str(img_path),
                conf_threshold=conf_threshold,
                review_floor=review_floor,
                iou_threshold=iou_threshold,
                weak_classes=weak_classes,
            )
        except Exception as exc:
            print(f"Hybrid failed: {exc}")
            hybrid_dets = []
        hybrid_time = time.perf_counter() - t0
        hybrid_latencies.append(hybrid_time)

        # Track class counts
        for d in yolo_dets:
            yolo_class_counts[d["item"]] = yolo_class_counts.get(d["item"], 0) + 1
        for d in hybrid_dets:
            hybrid_class_counts[d["item"]] = hybrid_class_counts.get(d["item"], 0) + 1
            if d.get("source") == "clip_fallback":
                clip_relabelled[d["item"]] = clip_relabelled.get(d["item"], 0) + 1

        clip_count = sum(1 for d in hybrid_dets if d.get("source") == "clip_fallback")
        per_image.append({
            "image": img_path.name,
            "yolo_count": len(yolo_dets),
            "hybrid_count": len(hybrid_dets),
            "clip_relabelled": clip_count,
            "yolo_time_ms": round(yolo_time * 1000, 1),
            "hybrid_time_ms": round(hybrid_time * 1000, 1),
        })

        print(f"YOLO={len(yolo_dets)} ({yolo_time*1000:.0f}ms)  "
              f"Hybrid={len(hybrid_dets)} ({hybrid_time*1000:.0f}ms)  "
              f"CLIP={clip_count}")

    # --- Aggregate ---
    avg_yolo_ms = sum(yolo_latencies) * 1000 / max(len(yolo_latencies), 1)
    avg_hybrid_ms = sum(hybrid_latencies) * 1000 / max(len(hybrid_latencies), 1)
    latency_overhead = avg_hybrid_ms - avg_yolo_ms

    # Per-class comparison for weak classes
    weak_class_comparison: list[dict[str, Any]] = []
    for cls in weak_classes:
        yolo_n = yolo_class_counts.get(cls, 0)
        hybrid_n = hybrid_class_counts.get(cls, 0)
        clip_n = clip_relabelled.get(cls, 0)
        if yolo_n > 0 or hybrid_n > 0:
            weak_class_comparison.append({
                "class": cls,
                "yolo_detections": yolo_n,
                "hybrid_detections": hybrid_n,
                "clip_relabelled": clip_n,
                "delta": hybrid_n - yolo_n,
            })

    # Determine which classes genuinely benefit
    classes_improved = [c for c in weak_class_comparison if c["delta"] > 0]
    classes_unchanged = [c for c in weak_class_comparison if c["delta"] == 0]
    classes_degraded = [c for c in weak_class_comparison if c["delta"] < 0]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "images_tested": len(images),
        "weak_classes_count": len(weak_classes),
        "latency": {
            "avg_yolo_ms": round(avg_yolo_ms, 1),
            "avg_hybrid_ms": round(avg_hybrid_ms, 1),
            "overhead_ms": round(latency_overhead, 1),
        },
        "weak_class_comparison": weak_class_comparison,
        "classes_improved": [c["class"] for c in classes_improved],
        "classes_unchanged": [c["class"] for c in classes_unchanged],
        "classes_degraded": [c["class"] for c in classes_degraded],
        "clip_relabelled_total": sum(clip_relabelled.values()),
        "per_image": per_image,
        "guidance": build_usage_guidance(
            classes_improved, latency_overhead, len(weak_classes)
        ),
    }

    return summary


def build_usage_guidance(
    classes_improved: list[dict],
    latency_overhead_ms: float,
    total_weak: int,
) -> dict[str, str]:
    """Build usage guidance for when to use hybrid vs. YOLOv8-only."""
    if len(classes_improved) == 0:
        recommendation = "yolo_only"
        rationale = (
            "CLIP fallback did not improve detection for any weak classes. "
            "The additional latency is not justified. Use YOLOv8-only."
        )
    elif latency_overhead_ms > 5000:
        recommendation = "selective_clip"
        rationale = (
            f"CLIP improved {len(classes_improved)} classes but adds "
            f"{latency_overhead_ms:.0f}ms overhead. Apply CLIP only to "
            f"the specific improved classes, not broadly."
        )
    else:
        recommendation = "hybrid"
        rationale = (
            f"CLIP improved {len(classes_improved)} classes with acceptable "
            f"latency overhead ({latency_overhead_ms:.0f}ms). Hybrid pipeline "
            f"is recommended for accuracy-critical scenarios."
        )

    return {
        "recommendation": recommendation,
        "rationale": rationale,
        "latency_acceptable": "yes" if latency_overhead_ms < 3000 else "marginal" if latency_overhead_ms < 5000 else "no",
    }


def save_evaluation(summary: dict[str, Any], output_dir: str) -> str:
    """Save CLIP evaluation results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "clip_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-image CSV
    csv_path = out / "clip_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "yolo_count", "hybrid_count", "clip_relabelled",
            "yolo_time_ms", "hybrid_time_ms",
        ])
        writer.writeheader()
        writer.writerows(summary["per_image"])

    # Weak-class comparison CSV
    if summary["weak_class_comparison"]:
        wc_path = out / "weak_class_comparison.csv"
        with open(wc_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "class", "yolo_detections", "hybrid_detections",
                "clip_relabelled", "delta",
            ])
            writer.writeheader()
            writer.writerows(summary["weak_class_comparison"])

    print(f"\n✓ CLIP evaluation saved to {out}")
    return str(json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[2]
    _TEST_DIR = _ROOT / "data" / "raw" / "test_fridges"
    _OUTPUT_DIR = _ROOT / "results" / "day13_clip_evaluation"

    print("=" * 70)
    print("Day 13 — CLIP Fallback Evaluation: YOLOv8-only vs. Hybrid")
    print("=" * 70)

    result = evaluate_clip_fallback(str(_TEST_DIR))
    save_evaluation(result, str(_OUTPUT_DIR))

    # Print summary
    lat = result["latency"]
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Avg YOLO latency:   {lat['avg_yolo_ms']:.1f}ms")
    print(f"  Avg Hybrid latency: {lat['avg_hybrid_ms']:.1f}ms")
    print(f"  Overhead:           {lat['overhead_ms']:.1f}ms")
    print(f"  CLIP relabelled:    {result['clip_relabelled_total']} detections total")
    print(f"  Classes improved:   {len(result['classes_improved'])}")
    print(f"  Classes degraded:   {len(result['classes_degraded'])}")
    print(f"\n  Guidance: {result['guidance']['recommendation']}")
    print(f"  {result['guidance']['rationale']}")
