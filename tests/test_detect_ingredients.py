from pathlib import Path

import pytest

from backend.modules.detect_ingredients import detect_ingredients


TEST_IMAGE = Path("data/raw/test_fridges")


def test_returns_list():
    image_path = TEST_IMAGE / "fridge_01.jpg"
    if not image_path.exists():
        pytest.skip("No sample fridge image available for inference test")

    result = detect_ingredients(str(image_path))
    assert isinstance(result, list)


def test_detection_structure():
    image_path = TEST_IMAGE / "fridge_01.jpg"
    if not image_path.exists():
        pytest.skip("No sample fridge image available for inference test")

    result = detect_ingredients(str(image_path))
    if result:
        assert "item" in result[0]
        assert "confidence" in result[0]
        assert "bbox" in result[0]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        detect_ingredients("data/raw/test_fridges/does_not_exist.jpg")
