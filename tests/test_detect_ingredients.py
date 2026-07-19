"""Tests for the updated detect_ingredients module (Day 12)."""

from pathlib import Path

import pytest

from backend.modules.detect_ingredients import detect_ingredients, get_model_info


TEST_IMAGE_DIR = Path("data/raw/test_fridges")


# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------

def test_model_info_returns_dict():
    info = get_model_info()
    assert isinstance(info, dict)
    assert "loaded" in info
    assert "source" in info


# ---------------------------------------------------------------------------
# Basic contract (carried forward from Day 11)
# ---------------------------------------------------------------------------

def _get_test_image() -> Path | None:
    """Return the first available test image, or None."""
    if not TEST_IMAGE_DIR.exists():
        return None
    for p in sorted(TEST_IMAGE_DIR.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return p
    return None


def test_returns_list():
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available for inference test")

    result = detect_ingredients(str(img))
    assert isinstance(result, list)


def test_detection_structure():
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available for inference test")

    result = detect_ingredients(str(img))
    if result:
        det = result[0]
        assert "item" in det
        assert "confidence" in det
        assert "bbox" in det
        assert "status" in det
        assert "source" in det


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        detect_ingredients("data/raw/test_fridges/does_not_exist.jpg")


# ---------------------------------------------------------------------------
# Day 12 — new parameters and fields
# ---------------------------------------------------------------------------

def test_status_field_values():
    """Detections should have status 'confirmed' or 'needs_review'."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    result = detect_ingredients(str(img), conf_threshold=0.25, review_floor=0.10)
    for det in result:
        assert det["status"] in ("confirmed", "needs_review")


def test_source_field_is_yolo():
    """Without CLIP fallback, source should always be 'yolo'."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    result = detect_ingredients(str(img))
    for det in result:
        assert det["source"] == "yolo"


def test_review_floor_captures_low_confidence():
    """With a very low floor, we should get at least as many detections
    as with the default floor."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    low_floor = detect_ingredients(str(img), conf_threshold=0.5, review_floor=0.05)
    high_floor = detect_ingredients(str(img), conf_threshold=0.5, review_floor=0.40)
    # Lower floor should capture at least as many detections
    assert len(low_floor) >= len(high_floor)


def test_iou_threshold_accepted():
    """Function should accept different IoU values without error."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    for iou in [0.3, 0.5, 0.7]:
        result = detect_ingredients(str(img), iou_threshold=iou)
        assert isinstance(result, list)


def test_confirmed_above_threshold():
    """Confirmed detections should have confidence >= conf_threshold."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    threshold = 0.25
    result = detect_ingredients(str(img), conf_threshold=threshold, review_floor=0.10)
    for det in result:
        if det["status"] == "confirmed":
            assert det["confidence"] >= threshold


def test_needs_review_below_threshold():
    """Needs-review detections should have confidence < conf_threshold."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    threshold = 0.25
    result = detect_ingredients(str(img), conf_threshold=threshold, review_floor=0.10)
    for det in result:
        if det["status"] == "needs_review":
            assert det["confidence"] < threshold


def test_lighting_enhancement_flag():
    """Function should accept use_lighting_enhancement without error."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    result = detect_ingredients(str(img), use_lighting_enhancement=True)
    assert isinstance(result, list)


def test_bbox_format():
    """Bounding boxes should be lists of 4 floats."""
    img = _get_test_image()
    if img is None:
        pytest.skip("No sample fridge image available")

    result = detect_ingredients(str(img))
    for det in result:
        assert len(det["bbox"]) == 4
        assert all(isinstance(x, float) for x in det["bbox"])
