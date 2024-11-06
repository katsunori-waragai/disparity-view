from pathlib import Path
from disparity_view.zed_info import CameraParameter

json_file = Path("../test/zed-imgs/camera_param.json")
camera_parameter = CameraParameter.load_json(json_file)
print(f"{camera_parameter=}")
print(f"{camera_parameter.to_matrix()=}")
print(f"{camera_parameter.get_baseline()=}")
