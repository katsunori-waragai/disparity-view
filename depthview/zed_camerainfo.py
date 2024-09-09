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
class CameraParameter:
    """
    camera_parameter = CameraParameter(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)
    print(camera_parameter.to_json())

    json_file = Path("tmp.json")
    camera_parameter.save_json(json_file)
    parameter = CameraParameter.load_json(json_file)

    camera_parameter2 = CameraParameter()
    camera_parameter2.get_current_setting(cam_info)

    camera_parameter3 = CameraParameter.create(cam_info)
    """

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
