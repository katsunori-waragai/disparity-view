import numpy as np

# -----------------------------
# 1. depth -> [0,1) 圧縮
# -----------------------------
def depth_to_unit(d: np.ndarray, lam=-3.0) -> np.ndarray:
    assert d.ndim == 2, f"{d.ndim} is not 2nd dimension"
    assert d.dtype in (np.float32, np.float64)
    d = np.maximum(d, 1e-8)
    return 1.0 - (1.0 + d)**lam   # Barron風の単調変換


# -----------------------------
# 2. [0,1) -> RGB（キューブエッジ）
# -----------------------------
def unit_to_rgb(t: np.ndarray) -> np.ndarray:
    """
    RGBキューブのエッジを辿るpiecewise線形マッピング
    """
    assert t.ndim == 2, f"{t.ndim} is not 2nd dimension"
    assert t.dtype in (np.float32, np.float64)
    t = np.clip(t, 0, 1 - 1e-8)

    # 6区間（RGB cube edges）
    segment = (t * 6).astype(int)
    local_t = t * 6 - segment

    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)

    # 各エッジ
    mask = segment == 0  # (0,0,0) -> (1,0,0)
    r[mask] = local_t[mask]

    mask = segment == 1  # (1,0,0) -> (1,1,0)
    r[mask] = 1
    g[mask] = local_t[mask]

    mask = segment == 2  # (1,1,0) -> (0,1,0)
    r[mask] = 1 - local_t[mask]
    g[mask] = 1

    mask = segment == 3  # (0,1,0) -> (0,1,1)
    g[mask] = 1
    b[mask] = local_t[mask]

    mask = segment == 4  # (0,1,1) -> (0,0,1)
    g[mask] = 1 - local_t[mask]
    b[mask] = 1

    mask = segment == 5  # (0,0,1) -> (0,0,0)
    b[mask] = 1 - local_t[mask]

    rgb = np.stack([r, g, b], axis=-1)
    assert rgb.ndim == 3, f"{rgb.ndim} is not 3nd dimension"
    return rgb


# -----------------------------
# フル変換
# -----------------------------
def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    t = depth_to_unit(depth)
    return unit_to_rgb(t)


import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
# -----------------------------
# デモ用 depth（擬似データ）
# -----------------------------
# h, w = 240, 320
# y = np.linspace(0, 5, h)\
# depth = np.tile(y[:, None], (1, w))

disparity=np.load(Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy"))

depth = 1 / disparity

depth *= 10.0

# -----------------------------
# 変換
# -----------------------------
rgb = depth_to_rgb(depth)

# -----------------------------
# 可視化
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Depth (raw)")
plt.imshow(depth, cmap='gray')
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Depth -> RGB (Vision Banana style)")
plt.imshow(rgb)

plt.tight_layout()
plt.show()
