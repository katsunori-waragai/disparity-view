from pathlib import Path

import numpy as np

from disparity_view.animation_gif import AnimationGif


def test_animation_gif():
    outname = Path("junk.gif")
    maker = AnimationGif()
    for i in range(10):
        image = np.full((256, 256), 10 * i, dtype=np.uint8)
        maker.append(image)
    maker.save(outname)

    assert outname.stat().st_size > 0


if __name__ == "__main__":
    test_animation_gif()
