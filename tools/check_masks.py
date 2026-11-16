"""
Candidate leaf mask for images that fail the uniform-background test (shaded or coloured paper), and a sheet to check it
by eye. Candidate: Otsu threshold on (R + G) / 2 - B, largest connected part, holes filled.

  python tools/check_masks.py   -> results/_analysis/mask_check.png, mask_check.json (coverage per class and dataset)
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from background_test import border, raw_image  # noqa: E402
from data import load_split  # noqa: E402
from utils.gradcam import leaf_mask  # noqa: E402

AREA = (0.02, 0.90)
MIN_SOLIDITY = 0.80


def otsu(v, bins=64):
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w = h.cumsum() / max(h.sum(), 1)
    m = (h * c).cumsum() / max(h.sum(), 1)
    between = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-9)
    return float(c[int(np.argmax(between))])


def candidate_mask(img):
    """boolean leaf mask, or None if the largest part touches most of the border or has an implausible area"""
    # leaves are green to brown (little blue); the paper is white, lilac or blue, also where it is shaded or brightly lit
    d = (img[..., 0] + img[..., 1]) / 2 - img[..., 2]
    d = ndimage.gaussian_filter(d, 1.0)
    # dark glossy parts of a leaf have little colour; a shadow on the paper stays lilac (more blue than green)
    lum = ndimage.gaussian_filter(img.mean(2), 1.0)
    blue_green = ndimage.gaussian_filter(img[..., 2] - img[..., 1], 1.0)
    m = (d > max(otsu(d), 12.0)) | ((lum < 90) & (blue_green < 8))
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = ndimage.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)
    edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
    if not (AREA[0] <= m.mean() <= AREA[1] and edge < 0.25) or (m.mean() < 0.05 and edge > 0):
        return None
    # a torn mask is not used: the leaf must fill most of its convex hull
    ys, xs = np.nonzero(m)
    return m if m.sum() / ConvexHull(np.column_stack([xs, ys])).volume >= MIN_SOLIDITY else None


def main():
    res, rows = {}, []
    for ds in ('TeaSickness', 'TeaLeafBD'):
        root, classes, splits, _ = load_split(ds, os.path.join(ROOT, 'datasets_prepared'))
        cov = {c: {'n': 0, 'current': 0, 'candidate': 0} for c in classes}
        for rel, c in splits['test']:
            img = raw_image(root, rel, 224)
            cur, new = leaf_mask(img), candidate_mask(img)
            cov[c]['n'] += 1
            cov[c]['current'] += cur is not None
            cov[c]['candidate'] += new is not None
            if ds == 'TeaSickness' and cur is None:
                rows.append((c, img, new))
        res[ds] = cov
        print(ds, json.dumps(cov), flush=True)
    with open(os.path.join(ROOT, 'results', '_analysis', 'mask_check.json'), 'w') as f:
        json.dump(res, f, indent=1)
    # sheet: Tea Sickness test images that fail the current mask, image and candidate side by side
    cols, tile = 8, 112
    n = len(rows)
    sheet = Image.new('RGB', (cols * tile, (n // (cols // 2) + 1) * tile), 'white')
    for i, (c, img, m) in enumerate(rows):
        a = Image.fromarray(img.astype(np.uint8)).resize((tile, tile))
        if m is None:
            b = np.full((224, 224, 3), 255, np.uint8)
            b[::8] = 200
        else:
            b = np.where(m[..., None], img, np.array([255, 0, 255], np.float32)).astype(np.uint8)
        r, q = i // (cols // 2), (i % (cols // 2)) * 2
        sheet.paste(a, (q * tile, r * tile))
        sheet.paste(Image.fromarray(b).resize((tile, tile)), ((q + 1) * tile, r * tile))
    sheet.save(os.path.join(ROOT, 'results', '_analysis', 'mask_check.png'))
    print('sheet images', n, flush=True)


if __name__ == '__main__':
    main()
