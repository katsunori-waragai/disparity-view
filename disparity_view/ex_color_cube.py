import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from color_cube import color_cube_mapping, depth_to_rgbcube


def dummy_depth_image():
    disparity = np.load(Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy"))

    depth = 1 / disparity

    depth *= 20
    print(f"{np.min(depth)=}")
    print(f"{np.max(depth)=}")
    # depth = depth ** 2
    # depth -= 0.1 * depth.min()
    depth *= 10.0
    return depth


def show_colormap():
    data = np.zeros((700, 700), dtype=np.float32)
    for i in range(700):
        data[i, :] = i / 700.0
    print(f"{np.max(data.flatten())=}")
    rgb2 = color_cube_mapping(data)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 1, 1)

    plt.title("RGB (Vision Banana style)")
    plt.imshow(rgb2)
    plt.ylim([700, 0])
    plt.savefig("color_cube_mapping.png")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "output"
    show_colormap()
    depth = dummy_depth_image()

    print(f"{np.min(depth)=}")
    print(f"{np.max(depth)=}")
    rgb = depth_to_rgbcube(depth)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Depth (raw)")
    plt.imshow(depth, cmap='gray')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Depth -> RGB (Vision Banana style)")
    plt.imshow(rgb)

    plt.tight_layout()
    png_name = output_dir / "depth_to_rgbcube.png"
    plt.savefig(png_name)

