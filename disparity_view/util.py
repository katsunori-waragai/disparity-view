import numpy as np
import open3d as o3d

def dummy_camera_matrix(image_shape, focal_length: float = 1070) -> np.ndarray:
    """
    return dummy camera matrix

    Note:
        If you change camera resolution, camera parameters also changes.
    """
    # approximation
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0

    fx = focal_length  # [pixel]
    fy = focal_length  # [pixel]

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return camera_matrix

def dummy_pihhole_camera_intrincic(image_shape, focal_length: float = 1070):
    height, width = image_shape[:2]
    cx = image_shape[1] / 2.0
    cy = image_shape[0] / 2.0
    o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=focal_length, fy=focal_length, cx=cx, cy=cy)

