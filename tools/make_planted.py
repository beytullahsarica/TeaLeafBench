"""
Positive control for the background test: a copy of the Tea Sickness Dataset whose background is a flat colour that
identifies the class. A model trained on it can use the background, so the swapped-background test must show that it
follows the colour; if it does not, the test cannot detect reliance on the background.

  python tools/make_planted.py      -> datasets_review/PlantedTS/{train,val,test}/<class>/*.jpg (masked images only)
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from background_test import dilate, raw_image  # noqa: E402
from data import load_split  # noqa: E402
from utils.gradcam import leaf_mask  # noqa: E402

COLOURS = [(220, 60, 60), (60, 180, 75), (70, 110, 230), (240, 200, 40), (160, 80, 200), (70, 200, 220), (240, 130, 40),
           (200, 200, 200)]


def main():
    root, classes, splits, _ = load_split('TeaSickness', os.path.join(ROOT, 'datasets_prepared'))
    out = os.path.join(ROOT, 'datasets_review', 'PlantedTS')
    for s in ('train', 'val', 'test'):
        n = 0
        for rel, c in splits[s]:
            img = raw_image(root, rel, 224)
            m = leaf_mask(img)
            if m is None:
                continue
            col = np.array(COLOURS[classes.index(c)], np.float32)
            res = np.where(dilate(m)[..., None], img, col).astype(np.uint8)
            d = os.path.join(out, s, c)
            os.makedirs(d, exist_ok=True)
            Image.fromarray(res).save(os.path.join(d, os.path.splitext(os.path.basename(rel))[0] + '.png'))
            n += 1
        print(s, n, flush=True)


if __name__ == '__main__':
    main()
