"""Inference wrapper for the fine-tuned ingredient detector.

Day 12 updates:
- Confidence threshold default changed from 0.3 → 0.25 (recall-first)
- Added review_floor parameter for low-confidence "needs_review" tagging
- Added iou_threshold parameter for NMS tuning
- Added optional CLAHE lighting enhancement
- Output dicts now include "status" and "source" fields

Packaging limitation (Strategy A — MVP):
    Packaged items (milk cartons, cheese blocks, boxed goods) are a known
    gap.  The fine-tuned model was not trained on packaging-content
    associations.  This is accepted as a known limitation for MVP — the
    Week 5 frontend will allow users to manually add packaged items the
    photo pipeline cannot see.  Strategy B (supplementary packaging
    classes + retrain) is documented as a possible future iteration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _ROOT / "models" / "checkpoints" / "yolov8n_fridge_v1" / "weights" / "best.pt"

# Fallback to COCO-pretrained weights when no fine-tuned checkpoint exists
_FALLBACK_MODEL_PATH = _ROOT / "yolov8n.pt"

if _MODEL_PATH.exists():
    _model = YOLO(str(_MODEL_PATH))
    _model_source = "fine-tuned"
elif _FALLBACK_MODEL_PATH.exists():
    _model = YOLO(str(_FALLBACK_MODEL_PATH))
    _model_source = "coco-pretrained-fallback"
else:
    _model = None
    _model_source = "unavailable"


def detect_ingredients(
    image_path: str,
    conf_threshold: float = 0.25,
    review_floor: float = 0.10,
    iou_threshold: float = 0.5,
    use_lighting_enhancement: bool = False,
) -> list[dict[str, Any]]:
    """Run ingredient detection on a single fridge or pantry image.

    Detections are split into two tiers based on confidence:

    * **confirmed** — confidence ≥ *conf_threshold*
    * **needs_review** — confidence between *review_floor* and
      *conf_threshold*.  These are low-confidence candidates surfaced
      for the user to confirm or dismiss in the Week 5 frontend rather
      than being silently dropped.

    Detections below *review_floor* are treated as noise and discarded.

    Args:
        image_path: Path to the input image.
        conf_threshold: Minimum confidence for a ``"confirmed"`` detection.
            Default ``0.25`` — chosen for a recall-first use case (grocery
            / meal planning) where missing a real item is costlier than
            showing a false positive the user can dismiss.
        review_floor: Minimum confidence to keep a detection at all.
            Detections between *review_floor* and *conf_threshold* are
            tagged ``"needs_review"`` rather than dropped.
        iou_threshold: IoU threshold for YOLOv8's Non-Maximum Suppression.
            Lower values merge more overlapping boxes (good for cluttered
            fridge shelves); higher values preserve more distinct boxes.
        use_lighting_enhancement: If ``True``, apply CLAHE lighting
            enhancement before inference (helps with dim fridge interiors).

    Returns:
        A list of dictionaries, each containing:

        * ``item`` — predicted class name
        * ``confidence`` — model confidence score (float, 0–1)
        * ``bbox`` — bounding box as ``[x1, y1, x2, y2]``
        * ``status`` — ``"confirmed"`` or ``"needs_review"``
        * ``source`` — ``"yolo"`` (detection source for pipeline tracking)

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If the detector model is unavailable.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if _model is None:
        raise RuntimeError(
            "Detection model is not available. Train or place a checkpoint at "
            f"{_MODEL_PATH} before calling detect_ingredients()."
        )

    # --- Optional lighting enhancement ---
    source_input: str | np.ndarray = image_path
    if use_lighting_enhancement:
        from backend.modules.lighting_preprocessing import enhance_lighting_array

        img = cv2.imread(image_path)
        if img is not None:
            source_input = enhance_lighting_array(img)

    # --- Run YOLOv8 inference ---
    # Use review_floor as the conf parameter so we get *all* detections
    # above the noise floor, then tier them ourselves.
    results = _model.predict(
        source=source_input,
        conf=review_floor,
        iou=iou_threshold,
        verbose=False,
    )

    detections: list[dict[str, Any]] = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append(
                {
                    "item": _model.names[class_id],
                    "confidence": confidence,
                    "bbox": [float(x) for x in box.xyxy[0]],
                    "status": "confirmed" if confidence >= conf_threshold else "needs_review",
                    "source": "yolo",
                }
            )

    return detections


def get_model_info() -> dict[str, Any]:
    """Return metadata about the currently loaded detection model.

    Useful for logging and evaluation scripts.
    """
    if _model is None:
        return {"loaded": False, "source": _model_source}

    return {
        "loaded": True,
        "source": _model_source,
        "model_path": str(_MODEL_PATH if _model_source == "fine-tuned" else _FALLBACK_MODEL_PATH),
        "class_count": len(_model.names),
        "class_names": list(_model.names.values()),
    }
