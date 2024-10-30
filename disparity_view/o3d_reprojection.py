from typing import Tuple

import numpy as np

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
