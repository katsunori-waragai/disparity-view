"""
module for Camera Parameter
"""

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path

import numpy as np


@dataclass_json
@dataclass
class CameraParameter:
    """
    camera_parameter = CameraParameter(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)
    print(camera_parameter.to_json())

    json_file = Path("tmp.json")
    camera_parameter.save_json(json_file)
    parameter = CameraParameter.load_json(json_file)

    camera_parameter3 = CameraParameter.create(cam_info)
    """

    width: int = 0  # [pixel]
    height: int = 0  # [pixel]
    fx: float = 0.0  # [pixel]
    fy: float = 0.0  # [pixel]
    cx: float = 0.0  # [pixel]
    cy: float = 0.0  # [pixel]
    baseline: float = 0.0  # [mm]

    def save_json(self, name: Path):
        open(name, "wt").write(self.to_json())

    @classmethod
    def load_json(cls, name: Path):  # from_json() に名前を変える
        return cls.from_json(open(name, "rt").read())

    @classmethod
    def create(cls, cam_info):
        """
        Note:
            cam_info = zed.get_camera_information()

        function to get zed2i camera by StereoLabs

        - Camera parameters are obtained by zed sdk.
        - Camera parameters are saved in /usr/local/zed/settings/SN*.conf
            The file format is toml.
        """
        left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
        width, height, fx, fy, cx, cy = (
            left_cam_params.image_size.width,
            left_cam_params.image_size.height,
            left_cam_params.fx,
            left_cam_params.fy,
            left_cam_params.cx,
            left_cam_params.cy,
        )
        baseline = cam_info.camera_configuration.calibration_parameters.get_camera_baseline()
        return cls(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)

    def to_matrix(self) -> np.ndarray:
        """
        return camera intrinsics matrix
        """
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def get_baseline(self):
        return self.baseline
