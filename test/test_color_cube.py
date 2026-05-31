import numpy as np
import pytest

# from disparity_view import color_cube
from disparity_view import color_cube_mapping, depth_to_unit, depth_to_rgbcube


def test_color_cube_mapping_basic():
    t = np.linspace(0, 1 - 1e-6, num=100, dtype=np.float32).reshape(10, 10)
    rgb = color_cube_mapping(t)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.float32
    assert np.all(rgb >= 0)
    assert np.all(rgb <= 1.0)


def test_color_cube_mapping_wrong_dtype():
    t_int = np.zeros((5, 5), dtype=np.int32)
    with pytest.raises(AssertionError):
        color_cube_mapping(t_int)


def test_depth_to_unit_positive_lam():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    u = depth_to_unit(depth, lam=3.0, c=2.0)
    assert u.ndim == 2
    assert u.dtype in (np.float32, np.float64)
    assert np.min(u) >= 0
    assert np.max(u) < 1


def test_depth_to_unit_wrong_inputs():
    with pytest.raises(AssertionError):
        depth_to_unit(np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(AssertionError):
        depth_to_unit(np.array([[1, 2]], dtype=np.int32))


