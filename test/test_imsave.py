from pathlib import Path

import numpy as np
import skimage.io

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

if __name__ == "__main__":
    test_imsave()
