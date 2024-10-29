from pathlib import Path

import open3d as o3d
import numpy as np
import skimage.io

tum_data = o3d.data.SampleTUMRGBDImage()
depth_path = tum_data.depth_path
color_path = tum_data.color_path


def test_t_point_cloud():
    """
    read depth, color from files, and reproject from constructed point cloud
    """
    print(f"{depth_path=}")
    print(f"{color_path=}")
    device = o3d.core.Device("CPU:0")
    depth = o3d.t.io.read_image(depth_path).to(device)
    color = o3d.t.io.read_image(color_path).to(device)

    width = color.columns
    height = color.rows

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, depth_scale=5000.0, depth_max=10.0)

    assert hasattr(pcd, "project_to_rgbd_image")
    assert isinstance(intrinsic, o3d.core.Tensor)
    rgbd_reproj = pcd.project_to_rgbd_image(width, height, intrinsic, depth_scale=5000.0, depth_max=10.0)

    assert hasattr(rgbd_reproj, "color")
    assert hasattr(rgbd_reproj, "depth")

    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())

    assert isinstance(color_legacy, np.ndarray)
    assert isinstance(depth_legacy, np.ndarray)

    assert color_legacy.shape[0] == height
    assert color_legacy.shape[1] == width
    assert color_legacy.dtype == np.float32

    assert depth_legacy.shape[0] == height
    assert depth_legacy.shape[1] == width
    assert depth_legacy.dtype == np.float32


if __name__ == "__main__":
    """
    pcd.project_to_rgbd_imageの使い方を確認するためのスクリプト
    """

    test_t_point_cloud()