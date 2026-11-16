"""
Does the background carry the class? Interventions on the test images of the two datasets photographed on paper
(TeaLeafBD, TeaSickness), inference only, with the trained checkpoints (no retraining).

  python tools/background_test.py                         # every base run of both datasets, all seeds
  python tools/background_test.py --seeds 0 --skip_done    # seed 0 only, runs without background_test.json

1. Border colour only (no deep model): a logistic-regression classifier on the median colour (and its spread) of
   the image border, trained on the training set and tested on the test set of the same split. Above-chance
   accuracy means that the background alone predicts the class.
     -> results/_analysis/border_colour.json (also the mean border colour per class and split)
2. Per run, on the test images with a leaf mask (uniform background, same mask as the leaf-focus score):
     original         the image as it is
     leaf_only        background pixels replaced by a neutral grey (128, 128, 128): the leaf alone
     background_only  leaf pixels (mask dilated by 3 px) replaced by the image's own background colour: no leaf
     swapped          background pixels replaced by the background colour of a test image of another class;
                      'follows' = share of predictions equal to that other class
   and per image the occlusion leaf share S (as in the paper), the leaf area share A, and the ceiling-aware index
   (S - A) / (S_max - A), where S_max is the largest share an 8x8 occlusion map can put on the leaf (all of it in
   the cell with the largest leaf fraction): 0 = no preference for the leaf, 1 = all evidence on the leaf. Negative
   probability drops are set to 0, as in the leaf-focus score.
     -> results/<run>/background_test.json, results/<run>/background_test_per_image.csv
The tables are written by tools/make_paper_tables.py (background.tex, leaf_focus.tex).
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import load_split  # noqa: E402
from models import data_config, get_model, normalize_name  # noqa: E402
from utils import load_checkpoint  # noqa: E402
from utils.gradcam import leaf_mask, occlusion_maps  # noqa: E402

DATASETS = ('TeaLeafBD', 'TeaSickness')
GREY = np.array([128, 128, 128], np.float32)
CONDITIONS = ('original', 'leaf_only', 'background_only', 'swapped')


def raw_image(root, rel, size):
    return np.asarray(Image.open(os.path.join(root, rel)).convert('RGB').resize((size, size), Image.BILINEAR),
                      dtype=np.float32)


def border(a, w=4):
    return np.concatenate([a[:w].reshape(-1, 3), a[-w:].reshape(-1, 3), a[:, :w].reshape(-1, 3),
                           a[:, -w:].reshape(-1, 3)])


def dilate(mask, r=3):
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (torch.nn.functional.max_pool2d(m, 2 * r + 1, stride=1, padding=r)[0, 0] > 0).numpy()


def metrics(pred, labels, n_classes):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    present = sorted(set(labels.tolist()))
    return {'accuracy': float((pred == labels).mean()),
            'macro_f1': float(f1_score(labels, pred, labels=present, average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, pred)), 'n': int(len(labels)),
            'n_classes': n_classes}


# ----------------------------------------------------------------------------- 1. border colour only
def border_colour(data_root, out_dir, size):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    res = {}
    for ds in DATASETS:
        root, classes, splits, _ = load_split(ds, data_root)
        feats, per_class = {}, {}
        for s in ('train', 'test'):
            X, y = [], []
            for rel, c in splits[s]:
                b = border(raw_image(root, rel, size))
                X.append(np.concatenate([np.median(b, 0), b.std(0)]))
                y.append(classes.index(c))
            feats[s] = (np.array(X), np.array(y))
            per_class[s] = {c: [round(float(v), 1) for v in np.array(X)[np.array(y) == k, :3].mean(0)]
                            for k, c in enumerate(classes) if (np.array(y) == k).any()}
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(*feats['train'])
        Xt, yt = feats['test']
        prior = np.bincount(feats['train'][1], minlength=len(classes)).argmax()
        res[ds] = {'classes': classes, 'border_only': metrics(clf.predict(Xt), yt, len(classes)),
                   'majority_class': metrics(np.full_like(yt, prior), yt, len(classes)),
                   'mean_border_rgb': per_class}
        print(f"[BORDER] {ds}: border colour only acc={res[ds]['border_only']['accuracy']:.3f} "
              f"macro-F1={res[ds]['border_only']['macro_f1']:.3f} (majority class acc="
              f"{res[ds]['majority_class']['accuracy']:.3f})")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'border_colour.json'), 'w') as f:
        json.dump(res, f, indent=1)


# ----------------------------------------------------------------------------- 2. interventions per run
@torch.no_grad()
def run(run_dir, device, size, batch_size):
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    a = cfg['args']
    data_root = a.get('data_root') or os.path.join(ROOT, 'datasets_prepared')
    root, classes, splits, _ = load_split(a['dataset'], data_root)
    model = get_model(a['model'], num_classes=len(classes), pretrained=True, img_size=a['img_size'])
    model.load_state_dict(load_checkpoint(os.path.join(run_dir, 'best_model.pth'))['model_state_dict'])
    model = model.to(device).eval()
    mean, std = (torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1).to(device) for v in data_config(model))

    items = []                                     # test images with a leaf mask
    for rel, c in splits['test']:
        img = raw_image(root, rel, size)
        m = leaf_mask(img)
        if m is not None:
            items.append((rel, classes.index(c), img, m, np.median(border(img), 0)))
    labels = np.array([x[1] for x in items])
    rng = np.random.default_rng(0)                 # donor of the swapped background: a test image of another class
    donor = np.array([rng.choice(np.nonzero(labels != y)[0]) for y in labels])

    def tensor(batch):
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device) / 255.0
        return (x - mean) / std

    pred = {k: [] for k in CONDITIONS}
    occ = []
    for s in range(0, len(items), batch_size):
        idx = range(s, min(len(items), s + batch_size))
        views = {k: [] for k in CONDITIONS}
        for i in idx:
            _, _, img, m, bg = items[i]
            md = dilate(m)[..., None]
            views['original'].append(img)
            views['leaf_only'].append(np.where(md, img, GREY))
            views['background_only'].append(np.where(md, bg, img))
            views['swapped'].append(np.where(md, img, items[donor[i]][4]))
        for k in CONDITIONS:
            pred[k].append(model(tensor(views[k])).float().argmax(1).cpu().numpy())
        x = tensor(views['original'])
        maps = occlusion_maps(model, x, torch.from_numpy(pred['original'][-1]).to(device), grid=8).numpy()
        for j, i in enumerate(idx):
            m = items[i][3].astype(np.float32)
            area = float(m.mean())
            cells = m.reshape(8, size // 8, 8, size // 8).mean((1, 3))      # leaf fraction of every grid cell
            if maps[j].sum() <= 0:
                occ.append((area, float(cells.max()), np.nan, np.nan))
                continue
            share = float((maps[j] * m).sum() / maps[j].sum())
            smax = float(cells.max())
            occ.append((area, smax, share, (share - area) / (smax - area) if smax > area else np.nan))
    pred = {k: np.concatenate(v) for k, v in pred.items()}
    out = {'run': os.path.basename(run_dir), 'dataset': a['dataset'], 'model': normalize_name(a['model']),
           'seed': a['seed'], 'n_test': len(splits['test']), 'n_masked': len(items),
           'conditions': {k: metrics(pred[k], labels, len(classes)) for k in CONDITIONS}}
    out['conditions']['swapped']['follows_background'] = float((pred['swapped'] == labels[donor]).mean())
    o = np.array(occ, dtype=np.float64)
    out['occlusion'] = {'area_share': float(np.nanmean(o[:, 0])), 'max_cell_share': float(np.nanmean(o[:, 1])),
                        'leaf_share': float(np.nanmean(o[:, 2])), 'leaf_share_median': float(np.nanmedian(o[:, 2])),
                        'focus_index': float(np.nanmean(o[:, 3])), 'focus_index_median': float(np.nanmedian(o[:, 3])),
                        'focus_index_q25': float(np.nanpercentile(o[:, 3], 25)),
                        'focus_index_q75': float(np.nanpercentile(o[:, 3], 75)),
                        'n': int(np.isfinite(o[:, 3]).sum())}
    with open(os.path.join(run_dir, 'background_test_per_image.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label', 'donor_label'] + [f'pred_{k}' for k in CONDITIONS] +
                   ['leaf_area_share', 'max_cell_leaf_share', 'occ_leaf_share', 'occ_focus_index'])
        for i, it in enumerate(items):
            w.writerow([it[0], classes[it[1]], classes[labels[donor[i]]]] + [classes[pred[k][i]] for k in CONDITIONS]
                       + [f'{v:.4f}' for v in o[i]])
    with open(os.path.join(run_dir, 'background_test.json'), 'w') as f:
        json.dump(out, f, indent=1)
    c = out['conditions']
    print(f"[BG] {out['run']}: n={len(items)}/{out['n_test']} acc original={c['original']['accuracy']:.3f} "
          f"leaf-only={c['leaf_only']['accuracy']:.3f} background-only={c['background_only']['accuracy']:.3f} "
          f"follows swapped background={c['swapped']['follows_background']:.3f} | occlusion share="
          f"{out['occlusion']['leaf_share']:.3f} area={out['occlusion']['area_share']:.3f} "
          f"index={out['occlusion']['focus_index']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    p.add_argument('--data_root', default=os.path.join(ROOT, 'datasets_prepared'))
    p.add_argument('--seeds', type=int, nargs='*', default=None, help='default: all seeds')
    p.add_argument('--skip_done', action='store_true', help='skip runs that already have background_test.json')
    p.add_argument('--no_border', action='store_true', help='skip the border-colour classifier')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not a.no_border:
        border_colour(a.data_root, os.path.join(a.results_dir, '_analysis'), a.img_size)
    runs = []
    for ds in DATASETS:
        for d in sorted(glob.glob(os.path.join(a.results_dir, f'{ds.lower()}_*_base_s*'))):
            seed = int(d.rsplit('_s', 1)[1])
            # finished runs only (test_results.json is written last): best_model.pth also exists during training
            if not os.path.exists(os.path.join(d, 'test_results.json')) or (a.seeds is not None and seed not in a.seeds):
                continue
            if a.skip_done and os.path.exists(os.path.join(d, 'background_test.json')):
                continue
            runs.append(d)
    print(f"{len(runs)} runs")
    for d in runs:
        t0 = time.time()
        try:
            run(d, device, a.img_size, a.batch_size)
        except Exception as e:  # noqa: BLE001  (one failing model must not stop the others)
            print(f"[BG][FAIL] {os.path.basename(d)}: {type(e).__name__}: {e}")
        print(f"     {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
