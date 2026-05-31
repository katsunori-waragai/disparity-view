"""
A module for converting depth data to RGB, inspired by Vision Banana

Key point:
To enable conversion from depth to RGB images and back from RGB images to depth,
it is essential to keep the images in float format as much as possible.


See Also:
Image Generators are Generalist Vision Learners
https://arxiv.org/html/2604.20329v1
"""
import numpy as np

def depth_to_unit(d: np.ndarray, lam=-3.0) -> np.ndarray:
    # -----------------------------
    # 1. depth -> [0,1)  compression
    # -----------------------------
    # d in [m]
    assert d.ndim == 2, f"{d.ndim} is not 2nd dimension"
    assert d.dtype in (np.float32, np.float64)
    d = np.maximum(d, 1e-8)
    return 1.0 - (1.0 + d/(lam))**lam


def color_cube_mapping(t: np.ndarray) -> np.ndarray:
    # -----------------------------
    #	Color mapping that maps continuous values in the range [0.0, 1.0]
    #	to the seven consecutive faces of a cube using a color cube.
    #	2. [0,1) -> RGB（cube edge）
    # -----------------------------
    assert t.ndim == 2, f"{t.ndim} is not 2nd dimension"
    assert t.dtype in (np.float32, np.float64)
    t = np.maximum(t, 0)

    # 7 edgs on cube edges（RGB cube edges）
    segment = (t * 7).astype(int)
    local_t = t * 7 - segment

    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)

    # each edge
    mask = segment == 0  # (0,0,0) -> (1,0,0)
    r[mask] = local_t[mask]

    mask = segment == 1  # (1,0,0) -> (1,1,0)
    r[mask] = 1
    g[mask] = local_t[mask]

    mask = segment == 2  # (1,1,0) -> (0,1,0)
    g[mask] = 1
    r[mask] = 1 - local_t[mask]

    mask = segment == 3  # (0,1,0) -> (0,1,1)
    r[mask] = 0
    g[mask] = 1
    b[mask] = local_t[mask]

    mask = segment == 4  # (0,1,1) -> (0,0,1)
    r[mask] = 0
    b[mask] = 1
    g[mask] = 1 - local_t[mask]

    mask = segment == 5  # (0,0,1) -> (1,0,1)
    r[mask] = local_t[mask]
    g[mask] = 0
    b[mask] = 1

    mask = segment == 6  # (1,0,1) -> (1,1,1)
    g[mask] = local_t[mask]
    b[mask] = 1
    r[mask] = 1

    rgb = np.stack([r, g, b], axis=-1)
    assert rgb.ndim == 3, f"{rgb.ndim} is not 3nd dimension"
    assert rgb.dtype in (np.float32, np.float64)
    return rgb


def depth_to_rgbcube(depth: np.ndarray) -> np.ndarray:
    t = depth_to_unit(depth, lam=-2)
    return color_cube_mapping(t)


