"""
導出済みの視差画像に基づいて、右カメラでの再投影画像を生成するサンプルスクリプト
"""

import numpy as np
import cv2

from disparity_view.reprojection import reproject_from_left_and_disparity


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


if __name__ == "__main__":
    """
    python3 reproject.py test/test-imgs/disparity-IGEV/left_motorcycle.npy test/test-imgs/left/left_motorcycle.png
    """
    from pathlib import Path
    import argparse
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
    reprojected_image = reproject_from_left_and_disparity(left_image, disparity, camera_matrix)
    outname = Path(args.outdir) / f"reproject_{left_name.stem}.png"
    cv2.imwrite(str(outname), reprojected_image)
    print(f"saved {outname}")