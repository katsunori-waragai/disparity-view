"""
module to get zed2i camera by StereoLabs

- Camera parameters are obtained by zed sdk.
- Camera parameters are saved in /usr/local/zed/settings/SN*.conf
    The file format is toml.
"""

import pyzed.sl as sl  # ZED-SDK
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path


def get_width_height_fx_fy_cx_cy(left_cam_params):
    """
    Note:
        left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    """
    return (
        left_cam_params.image_size.width,
        left_cam_params.image_size.height,
        left_cam_params.fx,
        left_cam_params.fy,
        left_cam_params.cx,
        left_cam_params.cy,
    )


def get_baseline(cam_info) -> float:
    """
    Note:
        cam_info = zed.get_camera_information()
    """
    return cam_info.camera_configuration.calibration_parameters.get_camera_baseline()


def load_settings():
    from pathlib import Path
    import toml

    tomlname = sorted(Path("/usr/local/lib/zed/settings").glob("SN*.conf"))[0]
    zed_settings = toml.load(tomlname)
    print(zed_settings["STEREO"])


@dataclass_json
@dataclass
class CameraParmeter:
    width: int = 0  # [pixel]
    height: int = 0  # [pixel]
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0  # [pixel]
    cy: float = 0.0  # [pixel]
    baseline: float = 0.0

    def get_current_setting(self, cam_info):
        """
        Note:
            cam_info = zed.get_camera_information()
        """
        left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
        self.width, self.height, self.fx, self.fy, self.cx, self.cy = get_width_height_fx_fy_cx_cy(left_cam_params)
        self.baseline = get_baseline(cam_info)

    def save_json(self, name: Path):
        open(name, "wt").write(self.to_json())

    @classmethod
    def load_json(cls, name: Path):
        return cls.from_json(open(name, "rt").read())

    @classmethod
    def create(cls, cam_info):
        """
        Note:
            cam_info = zed.get_camera_information()
        """
        left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
        width, height, fx, fy, cx, cy = get_width_height_fx_fy_cx_cy(left_cam_params)
        baseline = get_baseline(cam_info)
        return cls(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)


if __name__ == "__main__":
    import inspect

    zed = sl.Camera()

    init_params = sl.InitParameters()

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {status}")
        exit(1)

    cam_info = zed.get_camera_information()

    for k, v in inspect.getmembers(cam_info):
        print(k, v)

    print(f"{cam_info.camera_configuration=}")
    print(f"{cam_info.sensors_configuration=}")
    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    for k, v in inspect.getmembers(cam_info.camera_configuration):
        print(k, v)

    for k, v in inspect.getmembers(cam_info.sensors_configuration):
        print(k, v)

    for k, v in inspect.getmembers(cam_info.camera_configuration.calibration_parameters):
        print(k, v)

    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    right_cam_params = cam_info.camera_configuration.calibration_parameters.right_cam

    for k, v in inspect.getmembers(left_cam_params):
        print(k, v)

    for k, v in inspect.getmembers(right_cam_params):
        print(k, v)

    # Print some of the camera parameters
    print("Left Camera Parameters:")
    print(f"image_size: {left_cam_params.image_size.width} x {left_cam_params.image_size.height}")
    print(f"Focal Length (fx, fy): {left_cam_params.fx}, {left_cam_params.fy}")
    print(f"Principal Point (cx, cy): {left_cam_params.cx}, {left_cam_params.cy}")
    print(f"Distortion Coefficients: {left_cam_params.disto}")

    print("\nRight Camera Parameters:")
    print(f"image_size: {right_cam_params.image_size.width} x {right_cam_params.image_size.height}")
    print(f"Focal Length (fx, fy): {right_cam_params.fx}, {right_cam_params.fy}")
    print(f"Principal Point (cx, cy): {right_cam_params.cx}, {right_cam_params.cy}")
    print(f"Distortion Coefficients: {right_cam_params.disto}")
    print("\n")
    print(f"{cam_info.camera_configuration.calibration_parameters.get_camera_baseline()=}")

    print(f"{get_width_height_fx_fy_cx_cy(left_cam_params)=}")
    width, height, fx, fy, cx, cy = get_width_height_fx_fy_cx_cy(left_cam_params)
    print(f"{get_baseline(cam_info)}")
    baseline = get_baseline(cam_info)
    zed.close()

    camera_parameter = CameraParmeter(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)
    print(camera_parameter)
    print(camera_parameter.to_json())

    json_file = Path("tmp.json")
    camera_parameter.save_json(json_file)
    parameter = CameraParmeter.load_json(json_file)
    print(f"{parameter=}")

    camera_parameter2 = CameraParmeter()
    camera_parameter2.get_current_setting(cam_info)
    print(f"{camera_parameter2=}")

    camera_parameter3 = CameraParmeter.create(cam_info)
    print(f"{camera_parameter3=}")
