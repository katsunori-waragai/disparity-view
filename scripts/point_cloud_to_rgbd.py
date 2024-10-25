import open3d as o3d
import numpy as np
import skimage.io
import cv2

if __name__ == "__main__":
    """
    pcd.project_to_rgbd_imageの使い方を確認するためのスクリプト
    """
    device = o3d.core.Device("CPU:0")
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth = o3d.t.io.read_image(tum_data.depth_path).to(device)
    color = o3d.t.io.read_image(tum_data.color_path).to(device)

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, depth_scale=5000.0, depth_max=10.0)
    rgbd_reproj = pcd.project_to_rgbd_image(640, 480, intrinsic, depth_scale=5000.0, depth_max=10.0)

    color_legacy = np.asarray(rgbd_reproj.color.to_legacy())
    depth_legacy = np.asarray(rgbd_reproj.depth.to_legacy())
    print(f"{color_legacy.dtype=}")
    print(f"{depth_legacy.dtype=}")
    skimage.io.imsave("color.png", color_legacy)
    skimage.io.imsave("depth.png", depth_legacy)

    color_img = skimage.img_as_ubyte(color_legacy)
    cv2.imwrite("color_ubyte.png", color_img[:, :, ::-1])
    # depth_img = skimage.img_as_ubyte(depth_legacy)
    # cv2.imwrite("depth_ubyte.png", depth_img)
