import numpy as np
import cv2
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from disparity_view.util import dummy_camera_matrix

def generate_point_cloud(disparity_map: np.ndarray, left_image: np.ndarray, camera_matrix: np.ndarray, baseline: float):
    """
    視差マップと左カメラのRGB画像から点群データを生成する関数

    Args:
        disparity_map: 視差マップ (HxW)
        left_image: 左カメラのRGB画像 (HxWx3)
        camera_matrix: カメラの内部パラメータ
        baseline: 基線長

    Returns:
        point_cloud: 点群データ (Nx3)
        color: 点の色情報 (Nx3)
    """

    height, width = disparity_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # 視差から深度を計算
    depth = baseline * camera_matrix[0, 0] / (disparity_map + 1e-8)

    # カメラ座標系での3D座標を計算
    X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
    Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
    Z = depth

    # 点群データと色情報を生成
    point_cloud = np.stack((X, Y, Z), axis=2).reshape(-1, 3)
    color = left_image.reshape(-1, 3)

    return point_cloud, color


def reproject_point_cloud(
    point_cloud: np.ndarray,
    color: np.ndarray,
    right_camera_intrinsics: np.ndarray,
    rvec=np.eye(3, dtype=np.float64),
    tvec=np.zeros(3, dtype=np.float64),
) -> np.ndarray:
    """
    点群データを右カメラ視点に再投影する関数

    Args:
        point_cloud: 点群データ (Nx3 numpy array)
        color: 点の色情報 (Nx3 numpy array)
        right_camera_intrinsics: 右カメラの内部パラメータ
        tvec: transfer vector

    Returns:
        reprojected_image: 再投影画像
    """

    # カメラ座標系から画像座標系に変換 (投影)
    dtype = tvec.dtype
    points_2d, _ = cv2.projectPoints(
        point_cloud, rvec=rvec, tvec=tvec, cameraMatrix=right_camera_intrinsics, distCoeffs=np.zeros(5, dtype=dtype)
    )
    points_2d = np.int32(points_2d).reshape(-1, 2)

    # 再投影画像の作成
    img_w, img_h = 2 * right_camera_intrinsics[0][2], 2 * right_camera_intrinsics[1][2]
    reprojected_image = np.zeros((int(img_h), int(img_w), 3), dtype=np.uint8)

    assert reprojected_image.shape[2] == 3

    # 点を画像に描画
    for pt, c in zip(points_2d, color):
        x, y = pt[0], pt[1]  # points_2dの形状に合わせて修正
        if 0 <= x < img_w and 0 <= y < img_h:
            reprojected_image[y, x] = c.astype(np.uint8)

    return reprojected_image


def reproject_from_left_and_disparity(
    left_image: np.ndarray,
    disparity: np.ndarray,
    camera_matrix: np.ndarray,
    baseline: float,
    rvec=np.eye(3, dtype=np.float64),
    tvec=np.zeros(3, dtype=np.float64),
) -> np.ndarray:
    """
    Returns a reprojected image based on the left camera image, disparity image and camera parameters.

    Args:
        left_image：　left camera image
        disparity:  disparity image（raw data)
        baseline: baseline length
        tvec: transfer vector
    Returns:
        reprojected_image: reprojected image
    """

    right_camera_intrinsics = camera_matrix

    point_cloud, color = generate_point_cloud(disparity, left_image, camera_matrix, baseline)
    return reproject_point_cloud(point_cloud, color, right_camera_intrinsics, rvec=rvec, tvec=tvec)


def gen_right_image(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path, axis=0):
    """
    save reproject right image file

    Args:
        disparity: disparity image
        left_image:left camera image
        outdir: destination directory
        left_name: file name of the left camera image
    Returns：
        None
    """
    camera_matrix = dummy_camera_matrix(left_image.shape)
    baseline = 120.0  # [mm] dummy same to ZED2i
    if axis == 0:
        tvec = np.array((-baseline, 0.0, 0.0))
    elif axis == 1:
        tvec = np.array((0.0, -baseline, 0.0))
    elif axis == 2:
        tvec = np.array((0.0, 0.0,  -baseline))

    reprojected_image = reproject_from_left_and_disparity(
        left_image, disparity, camera_matrix, baseline=baseline, tvec=tvec
    )
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
        disparity: disparity image
        left_image:left camera image
        outdir: destination directory
        left_name: file name of the left camera image
    Returns：
        None
    """
    camera_matrix = dummy_camera_matrix(left_image.shape)
    baseline = 120.0  # [mm] same to zed2i
    right_camera_intrinsics = camera_matrix

    point_cloud, color = generate_point_cloud(disparity, left_image, camera_matrix, baseline)

    pictures = []
    n = 16
    for i in tqdm(range(n + 1)):
        tvec = np.array((-baseline * i / n, 0.0, 0.0))
        reprojected_image = reproject_point_cloud(point_cloud, color, right_camera_intrinsics, tvec=tvec)
        reprojected_image = cv2.cvtColor(reprojected_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(reprojected_image)
        pictures.append(pil_image)

    gifname = outdir / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    pil_images_to_gif_animation(pictures, gifname=gifname)
