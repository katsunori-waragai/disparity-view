import sys

import pyzed.sl as sl

import  disparity_view

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

def get_width_height(cam_info):
    left_cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    width = left_cam_params.image_size.width
    height = left_cam_params.image_size.height
    return width, height

if __name__ == "__main__":
    """
    ZED2iの現在のカメラの解像度に即したカメラパラメータをZED　SDKから取得してファイルに保存する。
    ZED SDKのインストール済みのマシンから、ZED2iにUSB接続していること。
    """
    from pathlib import Path
    cam_info = get_zed_camerainfo()
    camera_parameter = disparity_view.CameraParameter.create(cam_info)
    width, height = get_width_height(cam_info)
    outname = Path("out") / f"zed_{width}_{height}.json"
    outname.parent.mkdir(exist_ok=True, parents=True)
    camera_parameter.save_json(outname)