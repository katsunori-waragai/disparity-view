try:
    import pyzed.sl as sl

    no_zed_sdk = False
except ImportError:
    no_zed_sdk = True

import sys
import pytest
from pathlib import Path

import numpy as np

import disparity_view


@pytest.mark.skipif(no_zed_sdk, reason="ZED SDK(StereoLabs) is not installed.")
def get_zed_camerainfo():
    zed = sl.Camera()

    init_params = sl.InitParameters()

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        sys.exit(1)

    cam_info = zed.get_camera_information()
    zed.close()
    return cam_info


def test_camera_param_load():
    json_file = Path("zed-imgs/zed_camera_param.json")
    param = disparity_view.CameraParameter.load_json(json_file)
    assert isinstance(param, disparity_view.CameraParameter)


def test_camera_param_load_and_save():
    json_file = Path("zed-imgs/zed_camera_param.json")
    param = disparity_view.CameraParameter.load_json(json_file)
    assert isinstance(param, disparity_view.CameraParameter)

    out_json_file = Path("zed-imgs/zed_camera_param_test.json")
    param.save_json(out_json_file)

    reloaded_param = disparity_view.CameraParameter.load_json(out_json_file)

    assert reloaded_param.width == param.width
    assert reloaded_param.height == param.height
    assert reloaded_param.fx == param.fx
    assert reloaded_param.fy == param.fy
    assert reloaded_param.cx == param.cx
    assert reloaded_param.cy == param.cy
    assert reloaded_param.baseline == param.baseline


@pytest.mark.skipif(no_zed_sdk, reason="ZED SDK(StereoLabs) is not installed.")
def test_camera_param_create():
    cam_info = get_zed_camerainfo()

    camera_parameter = disparity_view.CameraParameter.create(cam_info)
    print(f"{camera_parameter=}")
    assert isinstance(camera_parameter.width, int)
    assert isinstance(camera_parameter.height, int)
    assert isinstance(camera_parameter.fx, float)
    assert isinstance(camera_parameter.fy, float)
    assert isinstance(camera_parameter.cx, float)
    assert isinstance(camera_parameter.cy, float)


def test_camera_param_create_to_marix():
    json_file = Path("zed-imgs/zed_camera_param.json")
    camera_parameter = disparity_view.CameraParameter.load_json(json_file)

    intrinsics = camera_parameter.to_matrix()
    assert isinstance(intrinsics, np.ndarray)
    assert intrinsics.shape == (3, 3)
    assert intrinsics[0, 0] == camera_parameter.fx
    assert intrinsics[1, 1] == camera_parameter.fy
    assert intrinsics[0, 1] == 0.0
    assert intrinsics[1, 0] == 0.0

    assert intrinsics.dtype in (np.float32, np.float64)
