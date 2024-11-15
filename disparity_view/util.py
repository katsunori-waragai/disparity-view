from pathlib import Path

import numpy as np
import open3d as o3d
import skimage


def create_camera_matrix(image_shape, focal_length: float = 1070.0) -> np.ndarray:
    """
    return camera matrix

    Note:
        If you change camera resolution, camera parameters also changes.
    """
    # approximation
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0

    fx = focal_length  # [pixel]
    fy = focal_length  # [pixel]

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def dummy_pinhole_camera_intrincic(image_shape, focal_length: float = 1070) -> o3d.camera.PinholeCameraIntrinsic:
    height, width = image_shape[:2]
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=focal_length, fy=focal_length, cx=cx, cy=cy)


def safer_imsave(p: Path, img: np.ndarray):
    int_type = (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)
    if img.dtype in int_type:
        skimage.io.imsave(str(p), img)
    elif str(p).find("depth") > -1:
        max_val = np.max(img.flatten())
        uint8_img = np.array(img * 255 / max_val, dtype=np.uint8)
        skimage.io.imsave(str(p), uint8_img)
    else:
        uint8_img = np.array(img * 255, dtype=np.uint8)
        skimage.io.imsave(str(p), uint8_img)
