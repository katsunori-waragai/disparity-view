from pathlib import Path
import disparity_view

json_file = Path("../test/zed-imgs/camera_param.json")
camera_parameter = disparity_view.CameraParameter.load_json(json_file)
print(f"{camera_parameter=}")
print(f"{camera_parameter.to_matrix()=}")
print(f"{camera_parameter.get_baseline()=}")
