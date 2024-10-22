try:
    import pyzed.sl as sl

    no_zed_sdk = False
except ImportError:
    no_zed_sdk = True

import sys
import pytest

import disparity_view


@pytest.mark.skipif(no_zed_sdk, reason="ZED SDK(StereoLabs) is not installed.")
def test_get_baseline():
    zed = sl.Camera()

    init_params = sl.InitParameters()

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        sys.exit(1)

    cam_info = zed.get_camera_information()
    baseline = disparity_view.get_baseline(cam_info)
    assert 110 < baseline < 130
    zed.close()


@pytest.mark.skipif(no_zed_sdk, reason="ZED SDK(StereoLabs) is not installed.")
def test_get_fx_fy_cx_cy():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        sys.exit(1)

    cam_info = zed.get_camera_information()

    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam

    width, height, fx, fy, cx, cy = disparity_view.get_width_height_fx_fy_cx_cy(left_cam_params)

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


@pytest.mark.skipif(no_zed_sdk, reason="ZED SDK(StereoLabs) is not installed.")
def test_camera_param_create():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        sys.exit(1)

    cam_info = zed.get_camera_information()
    camera_parameter = disparity_view.CameraParameter.create(cam_info)
    print(f"{camera_parameter=}")
    assert isinstance(camera_parameter.width, int)
    assert isinstance(camera_parameter.height, int)
    assert isinstance(camera_parameter.fx, float)
    assert isinstance(camera_parameter.fy, float)
    assert isinstance(camera_parameter.cx, float)
    assert isinstance(camera_parameter.cy, float)
    zed.close()
