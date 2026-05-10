import numpy as np

# -----------------------------
# デモ用 depth（擬似データ）
# -----------------------------
h, w = 240, 320
y = np.linspace(0, 5, h)
depth = np.tile(y[:, None], (1, w))

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
