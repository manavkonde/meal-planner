"""Inference wrapper for the fine-tuned ingredient detector."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _ROOT / "models" / "checkpoints" / "yolov8n_fridge_v1" / "weights" / "best.pt"
_model = YOLO(str(_MODEL_PATH)) if _MODEL_PATH.exists() else None


def detect_ingredients(image_path: str, conf_threshold: float = 0.3) -> list[dict[str, Any]]:
    """Run ingredient detection on a single fridge or pantry image.

    Args:
        image_path: Path to the input image.
        conf_threshold: Minimum confidence to include a detection.

    Returns:
        A list of dictionaries with item, confidence, and bbox.

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

    results = _model.predict(source=image_path, conf=conf_threshold, verbose=False)
    detections: list[dict[str, Any]] = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "item": _model.names[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0]],
                }
            )

    return detections
