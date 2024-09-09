import inspect
import pyzed.sl as sl

from depthview.zed_camerainfo import get_width_height_fx_fy_cx_cy, get_baseline


def test_get_baseline():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set up initial parameters for the camera
    init_params = sl.InitParameters()

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        exit(1)

    # Retrieve camera information
    cam_info = zed.get_camera_information()
    baseline = get_baseline(cam_info)
    assert 110 < baseline < 130
    zed.close()


def test_get_fx_fy_cx_cy():

    # Create a ZED camera object
    zed = sl.Camera()

    # Set up initial parameters forget_baseline(cam_info)()

    init_params = sl.InitParameters()
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        exit(1)

    # Retrieve camera information
    cam_info = zed.get_camera_information()

    # # Access left and right camera parameters
    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam

    width, height, fx, fy, cx, cy = get_width_height_fx_fy_cx_cy(left_cam_params)

    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(fx, float)
    assert isinstance(fy, float)
    assert isinstance(cx, float)
    assert isinstance(cy, float)

    assert fx > 0
    assert fy > 0
    assert cx > 0
    assert cy > 0

    zed.close()
