import numpy as np

# -----------------------------
# 1. depth -> [0,1) 圧縮
# -----------------------------
def depth_to_unit(d, lam=-3.0):
    d = np.maximum(d, 1e-8)
    return 1.0 - (1.0 + d)**lam   # Barron風の単調変換


# -----------------------------
# 2. [0,1) -> RGB（キューブエッジ）
# -----------------------------
def unit_to_rgb(t):
    """
    RGBキューブのエッジを辿るpiecewise線形マッピング
    """
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
    return rgb


# -----------------------------
# フル変換
# -----------------------------
def depth_to_rgb(depth):
    t = depth_to_unit(depth)
    return unit_to_rgb(t)