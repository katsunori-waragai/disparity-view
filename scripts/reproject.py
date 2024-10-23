"""
導出済みの視差画像に基づいて、右カメラでの再投影画像を生成するサンプルスクリプト
"""

from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

from disparity_view.reprojection import reproject_from_left_and_disparity, generate_point_cloud, reproject_point_cloud


def dummy_camera_matrix(image_shape) -> np.ndarray:
    # 近似値
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0

    # ダミー
    fx = 1070  # [pixel]
    fy = fx

    # カメラパラメータの設定
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return camera_matrix


def gen_right_image(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path):
    """
    save reproject right image file

    Args:
        disparity: 視差画像
        left_image: 左カメラ画像
        outdir: 保存先のディレクトリ
        left_name: 左カメラ画像ファイル名
    Returns：
        None
    """
    camera_matrix = dummy_camera_matrix(left_image.shape)
    baseline = 100.0  # [mm] dummy
    tvec = np.array((-baseline, 0.0, 0.0))
    reprojected_image = reproject_from_left_and_disparity(left_image, disparity, camera_matrix, baseline, tvec)
    outname = outdir / f"reproject_{left_name.stem}.png"
    outname.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(outname), reprojected_image)
    print(f"saved {outname}")


def pil_images_to_gif_animation(pictures, gifname="animation.gif"):
    """
    save animation gif file using PIL.Image

    pictures: List of PIL.Image
    """
    pictures[0].save(gifname, save_all=True, append_images=pictures[1:], optimize=False, duration=200, loop=0)


def make_animation_gif(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path):
    """
    save animation gif file

    Args:
        disparity: 視差画像
        left_image: 左カメラ画像
        outdir: 保存先のディレクトリ
        left_name: 左カメラ画像ファイル名
    Returns：
        None
    """
    camera_matrix = dummy_camera_matrix(left_image.shape)
    baseline = 100  # カメラ間の距離[m]
    right_camera_intrinsics = camera_matrix

    # 点群データの生成
    point_cloud, color = generate_point_cloud(disparity, left_image, camera_matrix, baseline)

    pictures = []
    n = 16
    for i in tqdm(range(n + 1)):
        tvec = np.array((-baseline * i / n, 0.0, 0.0))
        reprojected_image = reproject_point_cloud(point_cloud, color, right_camera_intrinsics, tvec)
        reprojected_image = cv2.cvtColor(reprojected_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(reprojected_image)
        pictures.append(pil_image)

    gifname = outdir / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    pil_images_to_gif_animation(pictures, gifname=gifname)


if __name__ == "__main__":
    """
    python3 reproject.py ../test/test-imgs/disparity-IGEV/left_motorcycle.npy ../test/test-imgs/left/left_motorcycle.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="reprojector")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--gif", action="store_true", help="git animation")
    parser.add_argument("--outdir", default="output", help="output folder")
    args = parser.parse_args()
    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    left_image = cv2.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    if args.gif:
        make_animation_gif(disparity, left_image, Path(args.outdir), left_name)
    else:
        gen_right_image(disparity, left_image, Path(args.outdir), left_name)
