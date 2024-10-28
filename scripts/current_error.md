python3 reproject_open3d.py 
/home/waragai/.local/lib/python3.8/site-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.1' currently installed).
  from pandas.core.computation.check import NUMEXPR_INSTALLED
np.max(depth.flatten())=7969.076
open3d_right_intrinsic=PinholeCameraIntrinsic with width = 1482 and height = 994.
Access intrinsics with intrinsic_matrix.
Traceback (most recent call last):
  File "reproject_open3d.py", line 111, in <module>
    rgbd_reproj = pcd.project_to_rgbd_image(shape[1], shape[0], intrinsic, depth_scale=5000.0, depth_max=10.0)
AttributeError: 'open3d.cpu.pybind.geometry.PointCloud' object has no attribute 'project_to_rgbd_image'
