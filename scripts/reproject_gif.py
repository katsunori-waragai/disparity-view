"""
導出済みの視差画像に基づいて、右カメラでの再投影画像を生成するサンプルスクリプト
"""

import numpy as np
import cv2

from disparity_view.reprojection import reproject_from_left_and_disparity
from disparity_view.reprojection import reproject_point_cloud, generate_point_cloud


def dummy_camera_matrix(image_shape) -> np.ndarray:
    # 近似値
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0

    # ダミー
    fx = 1070  # [mm]
    fy = fx

    # カメラパラメータの設定
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return camera_matrix


def pil_images_to_gif_animation(pictures, gifname="animation.gif"):
    pictures[0].save(gifname, save_all=True, append_images=pictures[1:], optimize=False, duration=200, loop=0)


if __name__ == "__main__":
    """
    python3 reproject.py test/test-imgs/disparity-IGEV/left_motorcycle.npy test/test-imgs/left/left_motorcycle.png
    """
    from pathlib import Path
    import PIL
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="reprojector")
    parser.add_argument("disparity", help="disparity npy file")
    parser.add_argument("left", help="left image file")
    parser.add_argument("--outdir", default="output", help="save colored or ply")
    args = parser.parse_args()
    disparity_name = Path(args.disparity)
    left_name = Path(args.left)
    left_image = cv2.imread(str(left_name))
    disparity = np.load(str(disparity_name))
    camera_matrix = dummy_camera_matrix(left_image.shape)

    baseline = 100  # カメラ間の距離[m]

    right_camera_intrinsics = camera_matrix

    # 点群データの生成
    point_cloud, color = generate_point_cloud(disparity, left_image, camera_matrix, baseline)

    pictures = []
    n = 16
    for i in tqdm(range(n + 1)):
        shift_ratio = i / n
        tvec = np.array((-baseline * shift_ratio, 0.0, 0.0))
        reprojected_image = reproject_point_cloud(point_cloud, color, right_camera_intrinsics, tvec)
        reprojected_image = cv2.cvtColor(reprojected_image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(reprojected_image)
        pictures.append(pil_image)

    gifname = Path(args.outdir) / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    pil_images_to_gif_animation(pictures, gifname=gifname)
