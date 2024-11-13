import sys

import pyzed.sl as sl

import disparity_view


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

def zed_camera_resolutions():
    import inspect
    return {k: v for k, v in inspect.getmembers(sl.RESOLUTION) if str(v).find("RESOLUTION") > -1 and k.find("__") == -1
    }


def change_camera_resolution(new_resolution):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = resolutions[new_resolution]
    status = zed.open(init_params)
    cam_info = zed.get_camera_information()
    camera_parameter = disparity_view.CameraParameter.create(cam_info)
    width, height = get_width_height(cam_info)
    outname = Path("out") / f"zed_{width}_{height}.json"
    outname.parent.mkdir(exist_ok=True, parents=True)
    camera_parameter.save_json(outname)
    print(f"saved {outname}")


if __name__ == "__main__":
    """
    ZED2iの現在のカメラの解像度に即したカメラパラメータをZED　SDKから取得してファイルに保存する。
    ZED SDKのインストール済みのマシンから、ZED2iにUSB接続していること。
    """
    import argparse
    import inspect
    from pathlib import Path

    parser = argparse.ArgumentParser("save zed camera parameter")
    parser.add_argument("--list", action="store_true", help="list resolutions")
    parser.add_argument("--save", action="store_true", help="save current resolution json")
    parser.add_argument("--new_resolution", help="change to new resolution")
    args = parser.parse_args()

    cam_info = get_zed_camerainfo()
    if args.save:
        camera_parameter = disparity_view.CameraParameter.create(cam_info)
        width, height = get_width_height(cam_info)
        outname = Path("out") / f"zed_{width}_{height}.json"
        outname.parent.mkdir(exist_ok=True, parents=True)
        camera_parameter.save_json(outname)
        print(f"saved {outname}")
        exit()

    resolutions = zed_camera_resolutions()
    if args.list:
        for k, v in resolutions.items():
            print(k, v)

        width = cam_info.camera_configuration.resolution.width
        height = cam_info.camera_configuration.resolution.height
        print("## current resolution")
        print(f"{width=} {height=}")
        exit()
    elif args.new_resolution:
        if args.new_resolution in resolutions.keys():
            change_camera_resolution(args.new_resolution)