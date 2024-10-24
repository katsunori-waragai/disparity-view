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


def test_reproject_from_left_and_disparity():
    imfile1 = "../test/test-imgs/left/left_motorcycle.png"
    left_image = cv2.imread(str(imfile1))
    disparity = np.load("../test/test-imgs/disparity-IGEV/left_motorcycle.npy")
    camera_matrix = dummy_camera_matrix(left_image.shape)

    baseline = 120.0  # [mm] dummy same to ZED2i
    tvec = np.array((-baseline, 0.0, 0.0))
    reprojected_image = reproject_from_left_and_disparity(
        left_image, disparity, camera_matrix, baseline=baseline, tvec=tvec
    )
    cv2.imwrite("reprojected.png", reprojected_image)
    assert reprojected_image.shape == left_image.shape
