import cv2

import disparity_view


def test_depth2normal():
    input_name = "assets/depth.png"
    converter = disparity_view.DepthToNormalMap()
    depth_map = cv2.imread(input_name, cv2.IMREAD_UNCHANGED)
    normal_bgr = converter.convert(depth_map)
    assert depth_map.shape[:2] == normal_bgr.shape[:2]
    assert normal_bgr.shape[2] == 3
