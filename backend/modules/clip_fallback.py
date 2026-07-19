"""Day 13 — CLIP zero-shot classification fallback for weak classes.

Provides a supplementary detection path using OpenAI's CLIP model for
classes or scenarios where the fine-tuned YOLOv8 model's accuracy is
insufficient.  CLIP is applied only to cropped image regions (from
YOLOv8's bounding boxes), not as a full-image detector.

Design decisions:
- CLIP is significantly slower than YOLOv8 per-inference — it is scoped
  narrowly to low-confidence and weak-class cases only.
- The model is lazy-loaded on first use to avoid slowing down imports
  when the fallback is not needed.
- Prompt templates ("a photo of {label}") are used for candidate labels
  as they consistently outperform bare label words in CLIP zero-shot.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

# Lazy-loaded globals
_clip_model = None
_clip_processor = None
_clip_loaded = False


def load_clip_model() -> tuple[Any, Any]:
    """Lazy-load the CLIP model and processor.

    Uses ``openai/clip-vit-base-patch32`` via HuggingFace transformers.
    The model is cached after first load.

    Returns:
        A tuple of (CLIPModel, CLIPProcessor).

    Raises:
        ImportError: If ``transformers`` or ``torch`` are not installed.
    """
    global _clip_model, _clip_processor, _clip_loaded

    if _clip_loaded:
        return _clip_model, _clip_processor

    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch  # noqa: F401 — needed by transformers at runtime
    except ImportError as exc:
        raise ImportError(
            "CLIP fallback requires 'transformers' and 'torch'. "
            "Install them with: pip install transformers torch"
        ) from exc

    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model: {model_name} ...")
    _clip_model = CLIPModel.from_pretrained(model_name)
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_loaded = True
    print("✓ CLIP model loaded")

    return _clip_model, _clip_processor


def clip_classify_crop(
    crop_img: Image.Image,
    candidate_labels: list[str],
    use_prompt_template: bool = True,
) -> dict[str, Any]:
    """Zero-shot classify a cropped image region against candidate labels.

    Args:
        crop_img: PIL Image of the cropped region.
        candidate_labels: Unified taxonomy class names to compare against.
        use_prompt_template: If True, wraps each label in
            ``"a photo of {label}"`` which improves CLIP accuracy.

    Returns:
        A dict with keys:

        * ``item`` — best-matching label
        * ``confidence`` — similarity score (0–1)
        * ``source`` — ``"clip_fallback"``
        * ``all_scores`` — dict mapping each candidate label to its score
    """
    import torch

    model, processor = load_clip_model()

    # Prompt-template wrapping (known to improve CLIP zero-shot accuracy)
    if use_prompt_template:
        text_inputs = [f"a photo of {label}" for label in candidate_labels]
    else:
        text_inputs = candidate_labels

    inputs = processor(
        text=text_inputs,
        images=crop_img,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_labels)
        probs = logits_per_image.softmax(dim=1)

    scores = probs[0].cpu().numpy()
    best_idx = int(scores.argmax())

    all_scores = {label: float(scores[i]) for i, label in enumerate(candidate_labels)}

    return {
        "item": candidate_labels[best_idx],
        "confidence": float(scores[best_idx]),
        "source": "clip_fallback",
        "all_scores": all_scores,
    }


def identify_weak_classes(
    ap_threshold: float = 0.3,
    evaluation_path: str | None = None,
) -> list[str]:
    """Identify classes with notably low AP from Day 10/12 evaluation data.

    If no evaluation data is available (e.g. no fine-tuned model yet),
    returns a hardcoded list of classes known to be weak from the Day 5
    baseline analysis.

    Args:
        ap_threshold: Classes with AP@0.5 below this are considered weak.
        evaluation_path: Path to a JSON file with per-class AP data.

    Returns:
        A list of weak class names.
    """
    if evaluation_path is not None:
        import json
        eval_path = Path(evaluation_path)
        if eval_path.exists():
            with open(eval_path, "r") as f:
                data = json.load(f)
            # Expect format: {"class_name": ap_value, ...}
            if isinstance(data, dict):
                return [cls for cls, ap in data.items() if ap < ap_threshold]

    # Fallback: classes known to be weak from Day 5 baseline
    # (87.5% of unified taxonomy is missing from COCO entirely)
    # These are classes that COCO cannot detect at all:
    weak_from_day5 = [
        "eggs", "milk", "yogurt", "bread",
        "tomato", "spinach", "onion", "bell_pepper",
        "cucumber", "potato", "garlic", "cauliflower",
        "cabbage", "zucchini", "corn", "eggplant",
        "apricot", "avocado", "blueberry", "cherry",
        "grape", "kiwi", "lemon", "lime",
        "mango", "peach", "pear", "plum",
        "pomegranate", "raspberry", "strawberry", "watermelon",
        "mushroom", "ginger", "beetroot",
    ]

    return weak_from_day5


def detect_ingredients_with_fallback(
    image_path: str,
    conf_threshold: float = 0.25,
    review_floor: float = 0.10,
    iou_threshold: float = 0.5,
    weak_classes: list[str] | None = None,
    use_lighting_enhancement: bool = False,
) -> list[dict[str, Any]]:
    """Hybrid detection pipeline: YOLOv8 primary + CLIP fallback.

    Runs YOLOv8 first, then applies CLIP zero-shot classification on:
    1. Detections tagged ``"needs_review"`` (low YOLO confidence)
    2. Detections whose class is in *weak_classes*

    If CLIP produces a higher-confidence classification than YOLO for a
    given crop, the detection is relabelled with the CLIP result.

    Args:
        image_path: Path to the input image.
        conf_threshold: Confidence threshold for YOLO confirmed detections.
        review_floor: Noise floor for YOLO needs_review detections.
        iou_threshold: IoU threshold for NMS.
        weak_classes: List of class names to route through CLIP.
            If None, uses ``identify_weak_classes()`` to determine them.
        use_lighting_enhancement: Whether to apply CLAHE before YOLO.

    Returns:
        A list of detection dicts, each containing item, confidence,
        bbox, status, and source (``"yolo"`` or ``"clip_fallback"``).

    Performance note:
        CLIP adds ~100-500ms per crop on CPU.  Apply narrowly to keep
        total latency acceptable.
    """
    from backend.modules.detect_ingredients import detect_ingredients

    if weak_classes is None:
        weak_classes = identify_weak_classes()

    # Step 1: Run YOLOv8 detection
    detections = detect_ingredients(
        image_path,
        conf_threshold=conf_threshold,
        review_floor=review_floor,
        iou_threshold=iou_threshold,
        use_lighting_enhancement=use_lighting_enhancement,
    )

    if not detections:
        return detections

    # Step 2: Load full image for cropping
    img = Image.open(image_path)

    # Get the full candidate label list from the YOLO model
    from backend.modules.detect_ingredients import _model
    if _model is not None:
        candidate_labels = list(_model.names.values())
    else:
        candidate_labels = weak_classes

    # Step 3: Apply CLIP fallback to qualifying detections
    final_detections: list[dict[str, Any]] = []
    clip_applied_count = 0

    for det in detections:
        should_apply_clip = (
            det["status"] == "needs_review"
            or det["item"] in weak_classes
        )

        if should_apply_clip:
            try:
                x1, y1, x2, y2 = det["bbox"]
                # Ensure crop coordinates are valid
                crop = img.crop((
                    max(0, int(x1)),
                    max(0, int(y1)),
                    min(img.width, int(x2)),
                    min(img.height, int(y2)),
                ))

                # Skip tiny crops that won't give meaningful CLIP results
                if crop.width < 10 or crop.height < 10:
                    det["source"] = "yolo"
                    final_detections.append(det)
                    continue

                clip_result = clip_classify_crop(crop, candidate_labels)
                clip_applied_count += 1

                # Prefer CLIP's relabelling if it's meaningfully more confident
                if clip_result["confidence"] > det["confidence"]:
                    det["item"] = clip_result["item"]
                    det["confidence"] = clip_result["confidence"]
                    det["source"] = "clip_fallback"
                    # Upgrade status if CLIP is confident enough
                    if det["confidence"] >= conf_threshold:
                        det["status"] = "confirmed"
                else:
                    det["source"] = "yolo"

            except Exception as exc:
                # If CLIP fails on a crop, keep the YOLO result
                det["source"] = "yolo"
                print(f"  ⚠ CLIP fallback failed for {det['item']}: {exc}")
        else:
            det["source"] = "yolo"

        final_detections.append(det)

    if clip_applied_count > 0:
        print(f"  CLIP fallback applied to {clip_applied_count}/{len(detections)} detections")

    return final_detections
