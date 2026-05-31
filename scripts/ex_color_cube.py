import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from disparity_view import color_cube_mapping, depth_to_rgbcube


def dummy_depth_image(npy_path : Path) -> np.ndarray:
    """
    Load a disparity image and convert it to depth.
    """
    disparity = np.load(npy_path)
    depth = 1.0 / disparity

    depth *= 20
    return depth


def show_colormap(outdir: Path):
    """
    [0.0, 1.0]の範囲の画像をマッピングする例
    """
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
    pngname = outdir/ "color_cube_mapping.png"
    plt.savefig(pngname)


def modified_dummy_depth_image(depth: np.ndarray) -> np.ndarray:
    """
    元画像の範囲では、色が変化しにくいので、平均値で正規化してから、さらに非線形変換をかける。
    """
    mean_depth = np.mean(depth.flatten())
    print(f"{mean_depth=}")
    depth /= mean_depth
    depth = depth**2.0
    depth *= 30
    return depth


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "output"
    # show_colormap(outdir=output_dir)

    npy_path =Path("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    depth = dummy_depth_image(npy_path=npy_path)

    depth = modified_dummy_depth_image(depth)

    print(f"{np.min(depth)=}")
    print(f"{np.max(depth)=}")


    rgb = depth_to_rgbcube(depth)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.jet()
    plt.imshow(depth)
    plt.title("Depth -> RGB (jet mapping)")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Depth -> RGB (Vision Banana style)")
    plt.imshow(rgb)

    plt.tight_layout()
    png_name = output_dir / "depth_to_rgbcube.png"
    plt.savefig(png_name)
    print(f"Saved: {png_name}")
