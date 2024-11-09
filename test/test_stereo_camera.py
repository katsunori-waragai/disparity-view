from pathlib import Path

import disparity_view
def _test_stereo_camera_create_from_camera_param():
    """
    python3で実行させてテストします。
    """
    json_file = Path("../test/zed-imgs/camera_param.json")
    cam_param = disparity_view.CameraParameter.load_json(json_file)

    assert hasattr(disparity_view, "StereoCamera")
    assert hasattr(disparity_view.StereoCamera, "create_from_camera_param")
    stereo_camera = disparity_view.StereoCamera.create_from_camera_param(cam_param)
    assert isinstance(stereo_camera, disparity_view.StereoCamera)
    assert stereo_camera.baseline > 0
    print(f"{stereo_camera.baseline=}")

if __name__ == "__main__":
    _test_stereo_camera_create_from_camera_param()
