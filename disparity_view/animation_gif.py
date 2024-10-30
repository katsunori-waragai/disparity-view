from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


@dataclass
class AnimationGif:
    """
    maker = AnimationGif()
    maker.append(image)
    maker.append(image)
    maker.save(gifname)
    """

    images: List = field(default_factory=list)

    def append(self, image: np.ndarray):
        "append image to list for animation gif"
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        self.images.append(pil_image)

    def save(self, gifname: Path):
        "save animation gif file using PIL.Image"
        print(len(self.images))
        self.images[0].save(gifname, save_all=True, append_images=self.images[1:], optimize=False, duration=200, loop=0)


if __name__ == "__main__":
    maker = AnimationGif()
    for i in range(10):
        image = np.full((256, 256), 10 * i, dtype=np.uint8)
        maker.append(image)
    maker.save("junk.gif")
