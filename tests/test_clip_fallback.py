"""Tests for the CLIP fallback module (Day 13).

Most tests that require the actual CLIP model are marked with
``pytest.mark.slow`` since model download + inference is heavy.
Tests that don't need the model test structural/utility logic only.
"""

from pathlib import Path

import pytest

from backend.modules.clip_fallback import identify_weak_classes


# ---------------------------------------------------------------------------
# identify_weak_classes (no model needed)
# ---------------------------------------------------------------------------

def test_identify_weak_classes_returns_list():
    result = identify_weak_classes()
    assert isinstance(result, list)
    assert len(result) > 0


def test_identify_weak_classes_contains_known_gaps():
    """Day 5 baseline identified these as completely missing from COCO."""
    result = identify_weak_classes()
    for expected in ["eggs", "milk", "tomato", "spinach", "potato"]:
        assert expected in result, f"{expected} should be in weak classes"


def test_identify_weak_classes_with_custom_file(tmp_path: Path):
    """When a per-class AP file is provided, it should use those values."""
    import json

    ap_data = {"apple": 0.8, "banana": 0.1, "tomato": 0.2}
    ap_path = tmp_path / "per_class_ap.json"
    with open(ap_path, "w") as f:
        json.dump(ap_data, f)

    result = identify_weak_classes(ap_threshold=0.3, evaluation_path=str(ap_path))
    assert "banana" in result
    assert "tomato" in result
    assert "apple" not in result


def test_identify_weak_classes_missing_file():
    """When evaluation_path doesn't exist, falls back to hardcoded list."""
    result = identify_weak_classes(evaluation_path="/nonexistent/path.json")
    assert isinstance(result, list)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# clip_classify_crop (requires CLIP model — marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_clip_classify_crop_structure():
    """CLIP classification should return expected dict structure."""
    from PIL import Image
    import numpy as np
    from backend.modules.clip_fallback import clip_classify_crop

    # Create a dummy image
    dummy = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
    labels = ["apple", "banana", "tomato"]

    result = clip_classify_crop(dummy, labels)
    assert "item" in result
    assert "confidence" in result
    assert "source" in result
    assert result["source"] == "clip_fallback"
    assert result["item"] in labels
    assert 0 <= result["confidence"] <= 1


@pytest.mark.slow
def test_clip_classify_crop_all_scores():
    """Should return scores for all candidate labels."""
    from PIL import Image
    import numpy as np
    from backend.modules.clip_fallback import clip_classify_crop

    dummy = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
    labels = ["apple", "carrot", "milk"]

    result = clip_classify_crop(dummy, labels)
    assert "all_scores" in result
    assert set(result["all_scores"].keys()) == set(labels)


# ---------------------------------------------------------------------------
# detect_ingredients_with_fallback (requires both YOLO and CLIP)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_detect_with_fallback_returns_list():
    """Hybrid detection should return a list of dicts."""
    from backend.modules.clip_fallback import detect_ingredients_with_fallback

    test_dir = Path("data/raw/test_fridges")
    images = sorted(test_dir.iterdir()) if test_dir.exists() else []
    if not images:
        pytest.skip("No test fridge images available")

    result = detect_ingredients_with_fallback(str(images[0]))
    assert isinstance(result, list)


@pytest.mark.slow
def test_detect_with_fallback_source_field():
    """Each detection should have source 'yolo' or 'clip_fallback'."""
    from backend.modules.clip_fallback import detect_ingredients_with_fallback

    test_dir = Path("data/raw/test_fridges")
    images = sorted(test_dir.iterdir()) if test_dir.exists() else []
    if not images:
        pytest.skip("No test fridge images available")

    result = detect_ingredients_with_fallback(str(images[0]))
    for det in result:
        assert det["source"] in ("yolo", "clip_fallback")


def test_detect_with_fallback_missing_file():
    """Should raise FileNotFoundError for missing image."""
    from backend.modules.clip_fallback import detect_ingredients_with_fallback

    with pytest.raises(FileNotFoundError):
        detect_ingredients_with_fallback("nonexistent_image.jpg")
