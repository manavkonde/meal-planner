"""Tests for the lighting preprocessing module."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.modules.lighting_preprocessing import (
    enhance_lighting,
    enhance_lighting_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    """Create a small dummy image for testing."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "test_img.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def dark_image(tmp_path: Path) -> Path:
    """Create a deliberately dark image to test enhancement."""
    img = np.random.randint(0, 50, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "dark_img.jpg"
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# enhance_lighting (file path variant)
# ---------------------------------------------------------------------------

def test_enhance_lighting_returns_ndarray(dummy_image: Path):
    result = enhance_lighting(str(dummy_image))
    assert isinstance(result, np.ndarray)


def test_enhance_lighting_preserves_shape(dummy_image: Path):
    original = cv2.imread(str(dummy_image))
    enhanced = enhance_lighting(str(dummy_image))
    assert enhanced.shape == original.shape


def test_enhance_lighting_preserves_dtype(dummy_image: Path):
    enhanced = enhance_lighting(str(dummy_image))
    assert enhanced.dtype == np.uint8


def test_enhance_lighting_missing_file():
    with pytest.raises(FileNotFoundError):
        enhance_lighting("nonexistent/path/image.jpg")


def test_enhance_lighting_dark_image_brightens(dark_image: Path):
    """Enhanced image should have higher mean luminance than original."""
    original = cv2.imread(str(dark_image))
    enhanced = enhance_lighting(str(dark_image))
    # Compare mean brightness in grayscale
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    assert enh_gray.mean() >= orig_gray.mean()


# ---------------------------------------------------------------------------
# enhance_lighting_array (in-memory variant)
# ---------------------------------------------------------------------------

def test_enhance_lighting_array_returns_ndarray():
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = enhance_lighting_array(img)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_enhance_lighting_array_with_custom_params():
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = enhance_lighting_array(img, clip_limit=3.0, tile_grid_size=(4, 4))
    assert result.shape == img.shape
