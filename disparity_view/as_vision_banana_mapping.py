"""
Vision Banana に即発された、深度をRGBに変換するライブラリ

重要なポイント：
深度からRGB画像、RGB画像から深度への再変換を可能にするためには、
極力floatのまま画像を保持することである。


See Also:
Image Generators are Generalist Vision Learners
https://arxiv.org/html/2604.20329v1
"""
import numpy as np

def depth_to_unit(d: np.ndarray, lam=-3.0) -> np.ndarray:
    # -----------------------------
    # 1. depth -> [0,1) 圧縮
    # -----------------------------
    assert d.ndim == 2, f"{d.ndim} is not 2nd dimension"
    assert d.dtype in (np.float32, np.float64)
    d = np.maximum(d, 1e-8)
    return 1.0 - (1.0 + d)**lam   # Barron風の単調変換


def unit_to_rgb(t: np.ndarray) -> np.ndarray:
    # -----------------------------
    # 2. [0,1) -> RGB（キューブエッジ）
    # -----------------------------
    """
    RGBキューブのエッジを辿るpiecewise線形マッピング
    """
    assert t.ndim == 2, f"{t.ndim} is not 2nd dimension"
    assert t.dtype in (np.float32, np.float64)
    t = np.maximum(t, 0)

    # 6区間（RGB cube edges）
    segment = (t * 7).astype(int)
    local_t = t * 7 - segment

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
    return rgb


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    t = depth_to_unit(depth, lam=-2)
    return unit_to_rgb(t)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from pathlib import Path

    disparity=np.load(Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy"))

    depth = 1 / disparity
    depth = depth**2
    depth -= 0.999 *depth.min()
    depth *= 10000.0

    print(f"{np.min(depth)=}")
    print(f"{np.max(depth)=}")
    rgb = depth_to_rgb(depth)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Depth (raw)")
    plt.imshow(depth, cmap='gray')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Depth -> RGB (Vision Banana style)")
    plt.imshow(rgb)

    plt.tight_layout()
    plt.savefig("depth_image.png")

    data= np.zeros((700, 700), dtype=np.float32)
    for i in range(700):
        data[:, i] = 1e-2 * i
    print(f"{np.max(data.flatten())=}")
    rgb2 = unit_to_rgb(data)
    plt.figure(figsize=(10,4))
    plt.subplot(1,1,1)

    plt.title("RGB (Vision Banana style)")
    plt.imshow(rgb2)
    plt.savefig("junk.png")