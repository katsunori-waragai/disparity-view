import open3d as o3d
import numpy as np

import inspect

tum_data = o3d.data.SampleTUMRGBDImage()
depth_path = tum_data.depth_path
color_path = tum_data.color_path

def test_t_rgbd_image():
    device = o3d.core.Device("CPU:0")
    depth = o3d.t.io.read_image(depth_path).to(device)
    color = o3d.t.io.read_image(color_path).to(device)

    width = color.columns
    height = color.rows

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    assert hasattr(rgbd, "color")
    assert hasattr(rgbd, "depth")
    assert hasattr(rgbd, "device")


def test_t_create_from_rgbd_image():
    device = o3d.core.Device("CPU:0")
    depth = o3d.t.io.read_image(depth_path).to(device)
    color = o3d.t.io.read_image(color_path).to(device)

    width = color.columns
    height = color.rows

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)


    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, depth_scale=5000.0, depth_max=10.0)
    assert hasattr(pcd, "project_to_rgbd_image")

    # このようにすると、pcdがどのようなメソッドを持っているのかが、わかる。
    for k, v in inspect.getmembers(pcd):
        if str(v).find("method") > -1:
            print(f"{k=} {v=}")

    rgbd_reproj = pcd.project_to_rgbd_image(width, height, intrinsic, depth_scale=5000.0, depth_max=10.0)

    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())

    assert isinstance(color_legacy, np.ndarray)
    assert isinstance(depth_legacy, np.ndarray)


if __name__ == "__main__":
    test_t_rgbd_image()
    test_t_create_from_rgbd_image()