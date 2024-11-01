from pathlib import Path

import numpy as np
import skimage.io

def safer_imsave(p: Path, img: np.ndarray):
    int_type = (np.uint8, np.uint16, np.uint32, np.uint64,
                np.int8, np.int16, np.int32, np.int64)
    if img.dtype in int_type:
        skimage.io.imsave(str(p), img)
    else:
        uint8_img = np.array(img * 255, dtype=np.uint8)
        skimage.io.imsave(str(p), uint8_img)

def test_imsave():
    left_name = Path("../test/test-imgs/left/left_motorcycle.png")
    img = skimage.io.imread(str(left_name))
    print(f"{np.max(img.flatten())=}")
    float_img = img.astype(dtype=np.float32) / 255.0
    print(f"{np.max(float_img.flatten())=}")

    outfile = Path("saved_float.png")
    skimage.io.imsave(str(outfile), float_img)
    assert outfile.lstat().st_size > 0

    img2 = skimage.io.imread(str(outfile))
    assert np.max(img2.flatten()) > 0

def test_safer_imsave():
    left_name = Path("../test/test-imgs/left/left_motorcycle.png")
    img = skimage.io.imread(str(left_name))
    print(f"{np.max(img.flatten())=}")
    float_img = img.astype(dtype=np.float32) / 255.0
    print(f"{np.max(float_img.flatten())=}")

    outfile = Path("saved_float.png")
    safer_imsave(str(outfile), float_img)
    assert outfile.lstat().st_size > 0

    img2 = skimage.io.imread(str(outfile))
    assert np.max(img2.flatten()) > 0

if __name__ == "__main__":
    test_imsave()
    test_safer_imsave()
