from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import skimage.io
import cv2

from tqdm import tqdm

from .animation_gif import AnimationGif


DEPTH_SCALE = 1000.0
DEPTH_MAX = 10.0


def shape_of(image) -> Tuple[float, float]:
    if isinstance(image, np.ndarray):
        return image.shape
    else:
        return (image.rows, image.columns)


def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    depth = baseline * float(focal_length) / (disparity + 1e-8)
    return depth


def dummy_o3d_camera_matrix(image_shape, focal_length: float = 535.4):
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0

    fx = focal_length  # [pixel]
    fy = focal_length  # [pixel]

    return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]


def as_extrinsics(tvec: np.ndarray, rot_mat=np.eye(3, dtype=float)) -> np.ndarray:
    return np.vstack((np.hstack((rot_mat, tvec.T)), [0, 0, 0, 1]))

def od3_reproject_point_cloud(pcd, intrinsics, tvec):
    extrinsics = as_extrinsics(tvec)
    img_w, img_h = int(2 * intrinsics[0][2]), int(2 * intrinsics[1][2])
    shape = [img_h, img_w]

    rgbd_reproj = pcd.project_to_rgbd_image(
        shape[1], shape[0], intrinsics=intrinsics, extrinsics=extrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )
    return rgbd_reproj


def o3d_reproject_from_left_and_disparity(left_image, disparity, intrinsics, baseline=120.0, tvec=np.array((0, 0, 0))):
    shape = left_image.shape
    focal_length = np.asarray(intrinsics)[0, 0]
    depth = disparity_to_depth(disparity, baseline, focal_length)

    open3d_img = o3d.t.geometry.Image(left_image)
    open3d_depth = o3d.t.geometry.Image(depth)

    rgbd = o3d.t.geometry.RGBDImage(open3d_img, open3d_depth)

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsics=intrinsics, depth_scale=DEPTH_SCALE, depth_max=DEPTH_MAX
    )


    rgbd_reproj = od3_reproject_point_cloud(pcd, intrinsics, tvec=tvec)
    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())

    return color_legacy, depth_legacy


def o3d_gen_right_image(disparity: np.ndarray, left_image: np.ndarray, outdir, left_name, axis):
    left_name = Path(left_name)
    shape = left_image.shape

    intrinsics = dummy_o3d_camera_matrix(shape, focal_length=535.4)
    baseline = 120  # カメラ間の距離[mm] 基線長

    scaled_baseline = baseline / DEPTH_SCALE
    if axis == 0:
        tvec = np.array([[-scaled_baseline, 0.0, 0.0]])
    elif axis == 1:
        tvec = np.array([[0.0, scaled_baseline, 0.0]])
    elif axis == 2:
        tvec = np.array([[0.0, 0.0, scaled_baseline]])

    color_legacy, depth_legacy = o3d_reproject_from_left_and_disparity(
        left_image, disparity, intrinsics, baseline=baseline, tvec=tvec
    )
    outdir.mkdir(exist_ok=True, parents=True)
    depth_out = outdir / f"depth_{left_name.stem}.png"
    color_out = outdir / f"color_{left_name.stem}.png"

    skimage.io.imsave(color_out, color_legacy)
    print(f"saved {color_out}")
    skimage.io.imsave(depth_out, depth_legacy)
    print(f"saved {depth_out}")


def make_animation_gif(disparity: np.ndarray, left_image: np.ndarray, outdir: Path, left_name: Path, axis=0):
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
    assert axis in (0, 1, 2)
    camera_matrix = dummy_o3d_camera_matrix(left_image.shape)
    baseline = 120.0  # [mm] same to zed2i

    point_cloud, color = od3_generate_point_cloud(disparity, left_image, camera_matrix, baseline)

    maker = AnimationGif()
    n = 16
    for i in tqdm(range(n + 1)):
        if axis == 0:
            tvec = np.array((-baseline * i / n, 0.0, 0.0))
        elif axis == 1:
            tvec = np.array((0.0, baseline * i / n, 0.0))
        elif axis == 2:
            tvec = np.array((0.0, 0.0, baseline * i / n))

        reprojected_image = od3_reproject_point_cloud(point_cloud, color, camera_matrix, tvec=tvec)
        maker.append(cv2.cvtColor(reprojected_image, cv2.COLOR_BGR2RGB))

    gifname = outdir / f"reproject_{left_name.stem}.gif"
    gifname.parent.mkdir(exist_ok=True, parents=True)
    maker.save(gifname)
