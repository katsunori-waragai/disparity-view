import open3d as o3d
import numpy as np


tum_data = o3d.data.SampleTUMRGBDImage()
depth_path = tum_data.depth_path
color_path = tum_data.color_path


def test_t_rgbd_image():
    device = o3d.core.Device("CPU:0")
    depth = o3d.t.io.read_image(depth_path).to(device)
    color = o3d.t.io.read_image(color_path).to(device)

    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    assert hasattr(rgbd, "color")
    assert hasattr(rgbd, "depth")
    assert hasattr(rgbd, "device")
    assert isinstance(rgbd, o3d.t.geometry.RGBDImage)


def test_t_create_from_rgbd_image():
    device = o3d.core.Device("CPU:0")
    depth = o3d.t.io.read_image(depth_path).to(device)
    color = o3d.t.io.read_image(color_path).to(device)

    width = color.columns
    height = color.rows

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)


    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, depth_scale=1000.0, depth_max=50.0)
    assert hasattr(pcd, "project_to_rgbd_image")

    rgbd_reproj = pcd.project_to_rgbd_image(width, height, intrinsic, depth_scale=1000.0, depth_max=50.0)

    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())

    assert isinstance(color_legacy, np.ndarray)
    assert isinstance(depth_legacy, np.ndarray)


def test_rgbd_image():
    depth = o3d.io.read_image(depth_path)
    color = o3d.io.read_image(color_path)

    assert hasattr(color, "columns") == False
    assert hasattr(color, "rows") == False

    """
    wrong code:
    No such constructor
    o3d.geometry.RGBDImage(color, depth)
    """
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)

    assert hasattr(rgbd, "color")
    assert hasattr(rgbd, "depth")


if __name__ == "__main__":
    test_t_rgbd_image()
    test_t_create_from_rgbd_image()
    test_rgbd_image()
