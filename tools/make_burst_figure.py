"""
Figure of burst leakage in TeaDisease: consecutive camera frames of the same leaf that the official split of the
dataset places in its training and its test set (the burst figure of the paper).

  python tools/make_burst_figure.py                                  # default pairs -> paper/figures/fig_bursts
  python tools/make_burst_figure.py --pairs black_rot:889:890 leaf_rust:780:781

The official split of every image is read from manifest.csv (column 'source': <train|valid|test>/images/...).
The pairs are chosen by hand: check by eye that both frames show the same leaf before using the figure.
"""
import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# standalone (numpy, Pillow, matplotlib; no torch): same style as utils/figures.py
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
NAMES = {'black_rot': 'Black Rot', 'healthy': 'Healthy', 'leaf_rust': 'Leaf Rust', 'red_spider': 'Red Spider Mite',
         'tea_mosquito_bug': 'Tea Mosquito Bug', 'white_spot': 'White Spot'}
OFFICIAL = {'train': 'train', 'valid': 'val', 'test': 'test'}
SET_NAME = {'train': 'training', 'val': 'validation', 'test': 'test'}
DEFAULT_PAIRS = ['black_rot:889:890', 'leaf_rust:780:781', 'healthy:520:521', 'white_spot:1000:1001']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(ROOT, 'datasets_prepared', 'TeaDisease'))
    p.add_argument('--pairs', nargs='+', default=DEFAULT_PAIRS, help='class:frame_a:frame_b')
    p.add_argument('--out', default=os.path.join(ROOT, 'paper', 'figures'))
    p.add_argument('--name', default='fig_bursts')
    a = p.parse_args()

    images = {}
    with open(os.path.join(a.data, 'manifest.csv')) as f:
        for r in csv.DictReader(f):
            m = re.search(r'IMG_(\d+)', os.path.basename(r['file']))
            images[(r['class'], int(m.group(1)))] = (os.path.join(a.data, r['file']),
                                                     OFFICIAL[r['source'].split('/')[0]], r['split'])
    pairs = []
    for s in a.pairs:
        c, fa, fb = s.split(':')
        pairs.append((c, int(fa), int(fb)))
    missing = [(c, f) for c, fa, fb in pairs for f in (fa, fb) if (c, f) not in images]
    if missing:
        sys.exit(f'not in {a.data}/manifest.csv: {missing}')

    # drawn at the printed size (full page width, 7.2 in): 2 pairs per row, labels 7 pt
    W, gap, pair_gap, label_h, head_h = 7.2, 0.04, 0.2, 0.28, 0.2
    per_row = 2
    panel = (W - per_row * gap - (per_row - 1) * pair_gap) / (2 * per_row)
    rows = (len(pairs) + per_row - 1) // per_row
    row_h = head_h + label_h + panel
    H = rows * row_h + 0.05
    fig = plt.figure(figsize=(W, H))
    names = {c: NAMES.get(c, c.replace('_', ' ')) for c, _, _ in pairs}
    for k, (c, fa, fb) in enumerate(pairs):
        row, col = divmod(k, per_row)
        x0 = col * (2 * panel + gap + pair_gap)
        y0 = H - (row + 1) * row_h                                      # bottom of the image panels
        fig.text((x0 + panel + gap / 2) / W, (y0 + panel + label_h + head_h / 2) / H,
                 f'{names[c]}: frames {fa} and {fb}', ha='center', va='center', fontsize=7.5, fontweight='bold')
        for j, fr in enumerate((fa, fb)):
            path, official, _ = images[(c, fr)]
            ax = fig.add_axes([(x0 + j * (panel + gap)) / W, y0 / H, panel / W, panel / H])
            im = Image.open(path).convert('RGB')
            ax.imshow(im.resize((600, 600), getattr(Image, 'Resampling', Image).LANCZOS))     # square, as the model sees it (no centre crop)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True), sp.set_linewidth(0.5)
            ax.set_title(f'IMG_{fr:04d}\nofficial {SET_NAME[official]} set',
                         fontsize=7, pad=2, linespacing=1.15)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, a.name)
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'-> {os.path.join(a.out, a.name)}.png / .pdf')
    for c, fa, fb in pairs:
        print(f'  {c}: IMG_{fa:04d} (official {images[(c, fa)][1]}, ours {images[(c, fa)][2]}) | '
              f'IMG_{fb:04d} (official {images[(c, fb)][1]}, ours {images[(c, fb)][2]})')


if __name__ == '__main__':
    main()
