"""
Inference-only analyses with the trained checkpoints (no retraining): cross-dataset, background, occlusion and box
tests. One json per task in results/_analysis/.

  python tools/review_interventions.py --task cross     cross-dataset test on masked leaves: grey background, normalised
                                                        leaf scale, background colours of the training dataset
  python tools/review_interventions.py --task bg        background controls: own-colour fill, swaps across colour groups,
                                                        group-level follow rate with a reference, follow rate among
                                                        changed predictions, colours by class, planted positive control
  python tools/review_interventions.py --task baselines model-free classifiers on the masked images only
  python tools/review_interventions.py --task occ       leaf-focus score: random-weights reference, other fill and grids,
                                                        absolute probability drops
  python tools/review_interventions.py --task box       field images: bounding boxes as the leaf region (box only, box
                                                        occluded, evidence inside the box), session and random split
--limit_runs / --limit_images make a small test run; --suffix keeps its output apart.
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
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from background_test import border, dilate, metrics, raw_image  # noqa: E402
from check_masks import candidate_mask  # noqa: E402
from data import load_split  # noqa: E402
from models import data_config, get_model, normalize_name  # noqa: E402
from utils import load_checkpoint  # noqa: E402
from utils.gradcam import leaf_mask  # noqa: E402

RES = os.path.join(ROOT, 'results')
PAPER = ('TeaLeafBD', 'TeaSickness')
ALIASES = {'tea_mosquito_bug': 'helopeltis'}
GREY = np.array([128, 128, 128], np.float32)
SIZE = 224
MASK_KIND = {"name": "current"}           # "extended": candidate mask (tools/check_masks.py) where the current mask fails


def done_runs(pattern, seeds=None):
    out = []
    for d in sorted(glob.glob(os.path.join(RES, pattern))):
        if os.path.exists(os.path.join(d, 'test_results.json')) and (seeds is None or int(d.rsplit('_s', 1)[1]) in seeds):
            out.append(d)
    return out


def load_run(run_dir, device, random_weights=False):
    with open(os.path.join(run_dir, 'config.json')) as f:
        a = json.load(f)['args']
    data_root = a.get('data_root') or os.path.join(ROOT, 'datasets_prepared')
    if not os.path.isabs(data_root):
        data_root = os.path.join(ROOT, data_root)
    root, classes, splits, _ = load_split(a['dataset'], data_root)
    model = get_model(a['model'], num_classes=len(classes), pretrained=not random_weights, img_size=a['img_size'])
    if not random_weights:
        model.load_state_dict(load_checkpoint(os.path.join(run_dir, 'best_model.pth'))['model_state_dict'])
    model = model.to(device).eval()
    mean, std = (torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1).to(device) for v in data_config(model))
    return a, root, classes, splits, model, mean, std


def masked_items(root, samples, limit=0):
    items = []
    if limit:                                      # small test runs: a mix of classes, not the first images of one class
        order = np.random.default_rng(0).permutation(len(samples))
        samples = [samples[int(i)] for i in order]
    for rel, c in samples:
        img = raw_image(root, rel, SIZE)
        m = leaf_mask(img)
        if m is None and MASK_KIND['name'] == 'extended':
            m = candidate_mask(img)
        if m is not None:
            items.append({'rel': rel, 'c': c, 'img': img, 'mask': dilate(m), 'raw_mask': m, 'bg': np.median(border(img), 0)})
            if limit and len(items) >= limit:
                break
    return items


@torch.no_grad()
def predict(model, views, mean, std, device, keep=None, bs=32):
    out = []
    for s in range(0, len(views), bs):
        x = torch.from_numpy(np.stack(views[s:s + bs])).permute(0, 3, 1, 2).float().to(device) / 255.0
        y = model((x - mean) / std).float()
        out.append(y[:, keep] if keep is not None else y)
    return torch.cat(out).argmax(1).cpu().numpy() if out else np.zeros(0, dtype=int)


def scaled_leaf(it, side=0.6):
    """leaf crop rescaled so that its longer side is `side` of the image, centred on the neutral grey"""
    ys, xs = np.nonzero(it['mask'])
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.where(it['mask'][..., None], it['img'], GREY)[y0:y1, x0:x1].astype(np.uint8)
    h, w = crop.shape[:2]
    s = side * SIZE / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    crop = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    out = np.tile(GREY, (SIZE, SIZE, 1))
    t, l = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[t:t + nh, l:l + nw] = crop
    return out


def save_json(res, name, suffix):
    out = os.path.join(RES, '_analysis', f'{name}{suffix}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


# ---------------------------------------------------------------------------------- cross-dataset on masked leaves
def task_cross(a, device):
    res = {}
    pairs = (('TeaLeafBD', 'TeaSickness'), ('TeaSickness', 'TeaLeafBD'))
    cache = {}

    def dataset_items(ds):
        if ds not in cache:
            root, classes, splits, _ = load_split(ds, os.path.join(ROOT, 'datasets_prepared'))
            cache[ds] = (classes, masked_items(root, splits['test'], a.limit_images))
        return cache[ds]
    for src, dst in pairs:
        src_classes, src_items = dataset_items(src)
        dst_classes, dst_items = dataset_items(dst)
        shared = [c for c in src_classes if c in dst_classes]
        keep = [src_classes.index(c) for c in shared]
        rng = np.random.default_rng(0)

        def views_of(items, donors):
            grey = [np.where(i['mask'][..., None], i['img'], GREY) for i in items]
            return {'original': [i['img'] for i in items], 'grey': grey, 'grey_scaled': [scaled_leaf(i) for i in items],
                    'train_background': [np.where(i['mask'][..., None], i['img'], donors[rng.integers(len(donors))]['bg'])
                                         for i in items]}
        cross_items = [i for i in dst_items if i['c'] in shared]
        own_items = [i for i in src_items if i['c'] in shared]
        cross_views, own_views = views_of(cross_items, src_items), views_of(own_items, src_items)
        cy = np.array([shared.index(i['c']) for i in cross_items])
        oy = np.array([shared.index(i['c']) for i in own_items])
        key = f'{src}->{dst}'
        res[key] = {'shared': shared, 'n_cross': len(cross_items), 'n_own': len(own_items),
                    'support_cross': np.bincount(cy, minlength=len(shared)).tolist(), 'runs': {}}
        for d in done_runs(f'{src.lower()}_*_224_base_s*', a.seeds)[:a.limit_runs or None]:
            t0 = time.time()
            try:
                _, _, _, _, model, mean, std = load_run(d, device)
                r = {}
                for name in cross_views:
                    r[f'cross_{name}'] = metrics(predict(model, cross_views[name], mean, std, device, keep), cy, len(shared))
                    r[f'own_{name}'] = metrics(predict(model, own_views[name], mean, std, device, keep), oy, len(shared))
                res[key]['runs'][os.path.basename(d)] = r
                print(f"[CROSS] {os.path.basename(d)} -> {dst}: " + ' '.join(
                    f"{k}={r['cross_' + k]['macro_f1']:.3f}/{r['own_' + k]['macro_f1']:.3f}" for k in cross_views) +
                    f" ({time.time() - t0:.0f}s)", flush=True)
                del model
            except Exception as e:  # noqa: BLE001
                print(f'[CROSS][FAIL] {d}: {type(e).__name__}: {e}', flush=True)
            save_json(res, 'review_cross_masked', a.suffix)


# ---------------------------------------------------------------------------------- background controls
def colour_groups(items, k=3):
    from sklearn.cluster import KMeans
    x = np.array([i['bg'] for i in items])
    return KMeans(n_clusters=min(k, len(x)), n_init=10, random_state=0).fit_predict(x)


def task_bg(a, device):
    res = {}
    targets = [(ds, f'{ds.lower()}_*_224_base_s*') for ds in PAPER] + [('PlantedTS', 'plantedts_*_224_planted_s*')]
    for ds, pat in targets:
        runs = done_runs(pat, a.seeds)[:a.limit_runs or None]
        if not runs:
            print(f'[BG] no runs for {ds}', flush=True)
            continue
        cfg, root, classes, splits, _, _, _ = load_run(runs[0], 'cpu', random_weights=True)
        items = masked_items(root, splits['test'], a.limit_images)
        y = np.array([classes.index(i['c']) for i in items])
        g = colour_groups(items)
        n_groups = int(g.max()) + 1
        # colour of each class and the colour group that holds most of its images
        class_group = np.array([np.bincount(g[y == k], minlength=n_groups).argmax() if (y == k).any() else -1
                                for k in range(len(classes))])
        rng = np.random.default_rng(0)
        donor_any = np.array([rng.choice(np.nonzero(y != v)[0]) for v in y])
        donor_grp = []
        for i, v in enumerate(y):
            c = np.nonzero((y != v) & (g != g[i]))[0]
            donor_grp.append(rng.choice(c) if len(c) else -1)
        donor_grp = np.array(donor_grp)
        has = donor_grp >= 0
        views = {'original': [i['img'] for i in items],
                 'leaf_only': [np.where(i['mask'][..., None], i['img'], GREY) for i in items],
                 'background_only': [np.where(i['mask'][..., None], i['bg'], i['img']) for i in items],
                 'own_fill': [np.where(i['mask'][..., None], i['img'], i['bg']) for i in items],
                 'swap_any': [np.where(i['mask'][..., None], i['img'], items[d]['bg']) for i, d in zip(items, donor_any)],
                 'swap_group': [np.where(i['mask'][..., None], i['img'], items[d]['bg'] if d >= 0 else i['bg'])
                                for i, d in zip(items, donor_grp)]}
        res[ds] = {'n': len(items), 'classes': classes, 'n_with_group_donor': int(has.sum()),
                   'colour_by_class': {c: {'mean_rgb': np.mean([i['bg'] for i in items if i['c'] == c], 0).round(1).tolist(),
                                           'n': int((y == k).sum()),
                                           'groups': np.bincount(g[y == k], minlength=n_groups).tolist()}
                                       for k, c in enumerate(classes) if (y == k).any()},
                   'group_mean_rgb': [np.mean([items[j]['bg'] for j in np.nonzero(g == k)[0]], 0).round(1).tolist()
                                      for k in range(n_groups)], 'runs': {}}
        for d in runs:
            t0 = time.time()
            try:
                _, _, _, _, model, mean, std = load_run(d, device)
                p = {k: predict(model, v, mean, std, device) for k, v in views.items()}
                r = {k: metrics(v, y, len(classes)) for k, v in p.items()}
                r['accuracy_by_class'] = {k: {c: float((v[y == n] == n).mean()) for n, c in enumerate(classes) if (y == n).any()} for k, v in p.items()}
                r['precision_by_class'] = {k: {c: float((y[v == n] == n).mean()) if (v == n).any() else None for n, c in enumerate(classes) if (y == n).any()} for k, v in p.items()}
                dg = y[donor_grp.clip(0)]
                group_of = class_group[p['swap_group']]
                ref_group = class_group[p['own_fill']]
                changed = (p['swap_group'] != p['own_fill']) & has
                r['follow'] = {
                    'class_any': float((p['swap_any'] == y[donor_any]).mean()),
                    'class_any_reference': float((p['own_fill'] == y[donor_any]).mean()),
                    'class_group': float((p['swap_group'][has] == dg[has]).mean()) if has.any() else None,
                    'class_group_reference': float((p['own_fill'][has] == dg[has]).mean()) if has.any() else None,
                    'colour_group': float((group_of[has] == g[donor_grp[has]]).mean()) if has.any() else None,
                    'colour_group_reference': float((ref_group[has] == g[donor_grp[has]]).mean()) if has.any() else None,
                    'share_changed': float(changed.sum() / max(has.sum(), 1)),
                    'class_among_changed': float((p['swap_group'][changed] == dg[changed]).mean()) if changed.any() else None,
                    'colour_group_among_changed': float((group_of[changed] == g[donor_grp[changed]]).mean()) if changed.any() else None}
                res[ds]['runs'][os.path.basename(d)] = r
                print(f"[BG] {os.path.basename(d)}: acc orig={r['original']['accuracy']:.3f} own={r['own_fill']['accuracy']:.3f} "
                      f"swap={r['swap_group']['accuracy']:.3f} follow={r['follow']['class_group']} "
                      f"ref={r['follow']['class_group_reference']} ({time.time() - t0:.0f}s)", flush=True)
                del model
            except Exception as e:  # noqa: BLE001
                print(f'[BG][FAIL] {d}: {type(e).__name__}: {e}', flush=True)
            save_json(res, 'review_bg', a.suffix)


def task_baselines(a, device):
    """border colour and leaf-pixel statistics, trained on the masked training images, tested on the masked test images"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    res = {}
    for ds in PAPER:
        root, classes, splits, _ = load_split(ds, os.path.join(ROOT, 'datasets_prepared'))
        tr = masked_items(root, splits['train'], a.limit_images)
        te = masked_items(root, splits['test'], a.limit_images)

        def feats(items, kind):
            out = []
            for i in items:
                if kind == 'border':
                    b = border(i['img'])
                    out.append(np.concatenate([np.median(b, 0), b.std(0)]))
                else:
                    v = i['img'][i['raw_mask']]
                    out.append(np.concatenate([v.mean(0), v.std(0), [i['raw_mask'].mean()]]))
            return np.array(out)
        ytr = np.array([classes.index(i['c']) for i in tr])
        yte = np.array([classes.index(i['c']) for i in te])
        r = {'n_train': len(tr), 'n_test': len(te), 'test_per_class': {c: int((yte == k).sum()) for k, c in enumerate(classes)},
             'train_per_class': {c: int((ytr == k).sum()) for k, c in enumerate(classes)},
             'majority_class': metrics(np.full_like(yte, np.bincount(ytr, minlength=len(classes)).argmax()), yte, len(classes))}
        for kind in ('border', 'leaf_pixels'):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            clf.fit(feats(tr, kind), ytr)
            r[kind] = metrics(clf.predict(feats(te, kind)), yte, len(classes))
        res[ds] = r
        print(f"[BASE] {ds}: {json.dumps(r)[:400]}", flush=True)
    save_json(res, 'review_baselines', a.suffix)


# ---------------------------------------------------------------------------------- occlusion: score validation
@torch.no_grad()
def occlusion(model, x, cls, fill, grid):
    """absolute probability drops per cell (not clamped, not normalised): (B, grid, grid), and the base probability"""
    H, W = x.shape[-2:]
    ys, xs = np.linspace(0, H, grid + 1).astype(int), np.linspace(0, W, grid + 1).astype(int)

    def prob(inp):
        with torch.autocast('cuda', dtype=torch.float16, enabled=inp.is_cuda):
            return model(inp).float().softmax(1).gather(1, cls[:, None])[:, 0]
    base = prob(x)
    drop = torch.zeros(x.shape[0], grid, grid, device=x.device)
    for r in range(grid):
        for c in range(grid):
            xo = x.clone()
            xo[:, 0, ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = fill[0]
            xo[:, 1, ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = fill[1]
            xo[:, 2, ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = fill[2]
            drop[:, r, c] = base - prob(xo)
    return drop.cpu().numpy(), base.cpu().numpy()


def focus_scores(drops, masks, grid):
    """leaf share S, leaf area A, S_max at this grid, index (S - A) / (S_max - A), absolute drops"""
    out = []
    for d, m in zip(drops, masks):
        cells = np.stack([[m[np.linspace(0, SIZE, grid + 1).astype(int)[r]:np.linspace(0, SIZE, grid + 1).astype(int)[r + 1],
                             np.linspace(0, SIZE, grid + 1).astype(int)[c]:np.linspace(0, SIZE, grid + 1).astype(int)[c + 1]].mean()
                           for c in range(grid)] for r in range(grid)])
        p = np.clip(d, 0, None)
        area, smax = float(m.mean()), float(cells.max())
        if p.sum() <= 0:
            out.append((area, smax, np.nan, np.nan, float(np.clip(d, 0, None).max()), float(p.sum())))
            continue
        s = float((p * cells).sum() / p.sum())
        out.append((area, smax, s, (s - area) / (smax - area) if smax > area else np.nan, float(p.max()), float(p.sum())))
    return np.array(out, dtype=np.float64)


def summarise(o):
    return {'leaf_share': float(np.nanmean(o[:, 2])), 'index': float(np.nanmean(o[:, 3])),
            'index_q25_q75': [float(np.nanpercentile(o[:, 3], 25)), float(np.nanpercentile(o[:, 3], 75))],
            'index_sd_images': float(np.nanstd(o[:, 3])), 'max_abs_drop': float(np.nanmean(o[:, 4])),
            'sum_abs_drop': float(np.nanmean(o[:, 5])), 'empty_maps': float(np.isnan(o[:, 2]).mean()), 'n': int(len(o))}


def task_whitespot(a, device):
    """Tea Sickness models on the TeaDisease images of White Spot (excluded from the splits there) and Tea leaf (read as
    healthy), outputs restricted to these two classes: the only disease class shared between paper and field images."""
    src = os.path.join(ROOT, 'datasets', 'tea_disease_dataset')
    root_p = os.path.join(ROOT, 'datasets_prepared', 'TeaDisease')
    with open(os.path.join(root_p, 'removed.csv'), newline='') as f:
        white = sorted({r['source'] for r in csv.DictReader(f) if r['class'] == 'white_spot' and 'shooting session' in r['reason']})
    with open(os.path.join(root_p, 'manifest.csv'), newline='') as f:
        healthy = [r['file'] for r in csv.DictReader(f) if r['class'] == 'healthy']
    # one image per distinct photograph: the release stores most photographs several times
    seen, w = set(), []
    for s in white:
        k = os.path.basename(s).split('.rf.')[0]
        if k not in seen:
            seen.add(k)
            w.append(s)
    items = [raw_image(src, s, SIZE) for s in w] + [raw_image(root_p, s, SIZE) for s in healthy]
    y = np.array([0] * len(w) + [1] * len(healthy))
    res = {'n_white_spot': len(w), 'n_healthy': len(healthy), 'runs': {}}
    for d in done_runs('teasickness_*_224_base_s*', a.seeds)[:a.limit_runs or None]:
        try:
            _, root, classes, splits, model, mean, std = load_run(d, device)
            keep = [classes.index('white_spot'), classes.index('healthy')]
            p = predict(model, items, mean, std, device, keep)
            allp = predict(model, items, mean, std, device)
            test = [(rel, c) for rel, c in splits['test'] if c in ('white_spot', 'healthy')]
            own = predict(model, [raw_image(root, rel, SIZE) for rel, _ in test], mean, std, device, keep)
            oy = np.array([0 if c == 'white_spot' else 1 for _, c in test])
            r = {'cross': metrics(p, y, 2), 'own': metrics(own, oy, 2),
                 'white_spot_recall': float((p[y == 0] == 0).mean()), 'healthy_recall': float((p[y == 1] == 1).mean()),
                 'white_spot_recall_all_outputs': float((allp[y == 0] == keep[0]).mean()),
                 'healthy_recall_all_outputs': float((allp[y == 1] == keep[1]).mean())}
            res['runs'][os.path.basename(d)] = r
            print(f"[WS] {os.path.basename(d)}: F1 {r['cross']['macro_f1']:.3f} own {r['own']['macro_f1']:.3f} white {r['white_spot_recall']:.3f} healthy {r['healthy_recall']:.3f}", flush=True)
            del model
        except Exception as e:  # noqa: BLE001
            print(f'[WS][FAIL] {d}: {type(e).__name__}: {e}', flush=True)
        save_json(res, 'review_whitespot', a.suffix)


def task_occ(a, device):
    res = {}
    for ds in PAPER:
        res[ds] = {}
        for d in done_runs(f'{ds.lower()}_*_224_base_s0')[:a.limit_runs or None]:
            t0 = time.time()
            try:
                _, root, classes, splits, model, mean, std = load_run(d, device)
                items = masked_items(root, splits['test'], a.limit_images or 200)
                masks = [i['raw_mask'].astype(np.float32) for i in items]
                zero = torch.zeros(3).to(device)
                grey = ((torch.tensor(GREY).to(device) / 255.0).view(1, 3, 1, 1) - mean) / std
                grey = grey.view(3)

                def run_model(m, fill, grid):
                    out = []
                    for s in range(0, len(items), 16):
                        x = torch.from_numpy(np.stack([i['img'] for i in items[s:s + 16]])).permute(0, 3, 1, 2).float().to(device) / 255.0
                        x = (x - mean) / std
                        with torch.no_grad():
                            cls = m(x).float().argmax(1)
                        dr, _ = occlusion(m, x, cls, fill, grid)
                        out.append(focus_scores(dr, masks[s:s + 16], grid))
                    return summarise(np.concatenate(out))
                r = {'trained_mean_fill_grid8': run_model(model, zero, 8),
                     'trained_grey_fill_grid8': run_model(model, grey, 8),
                     'trained_mean_fill_grid4': run_model(model, zero, 4),
                     'trained_mean_fill_grid16': run_model(model, zero, 16)}
                del model
                torch.manual_seed(0)
                _, _, _, _, rnd, _, _ = load_run(d, device, random_weights=True)
                r['random_weights_mean_fill_grid8'] = run_model(rnd, zero, 8)
                del rnd
                res[ds][os.path.basename(d)] = r
                print(f"[OCC] {os.path.basename(d)}: index trained={r['trained_mean_fill_grid8']['index']:.3f} grey="
                      f"{r['trained_grey_fill_grid8']['index']:.3f} g4={r['trained_mean_fill_grid4']['index']:.3f} g16="
                      f"{r['trained_mean_fill_grid16']['index']:.3f} random={r['random_weights_mean_fill_grid8']['index']:.3f} "
                      f"drop trained={r['trained_mean_fill_grid8']['sum_abs_drop']:.3f} random="
                      f"{r['random_weights_mean_fill_grid8']['sum_abs_drop']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f'[OCC][FAIL] {d}: {type(e).__name__}: {e}', flush=True)
            save_json(res, 'review_occlusion', a.suffix)


# ---------------------------------------------------------------------------------- boxes on the field photographs
def boxes_for(manifest_row):
    """YOLO boxes (normalised cx cy w h) of a kept image, from the label file of its source"""
    src = manifest_row['source']
    split = src.split('/')[0]
    stem = os.path.splitext(os.path.basename(src))[0]
    path = os.path.join(ROOT, 'datasets', 'tea_disease_dataset', split, 'labels', f'{stem}.txt')
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                v = line.split()
                if len(v) >= 5:
                    out.append(tuple(float(t) for t in v[1:5]))
    return out


def box_mask(boxes):
    m = np.zeros((SIZE, SIZE), dtype=bool)
    for cx, cy, w, h in boxes:
        x0, x1 = int((cx - w / 2) * SIZE), int(np.ceil((cx + w / 2) * SIZE))
        y0, y1 = int((cy - h / 2) * SIZE), int(np.ceil((cy + h / 2) * SIZE))
        m[max(0, y0):min(SIZE, y1), max(0, x0):min(SIZE, x1)] = True
    return m


def task_box(a, device):
    res = {}
    for tag, droot in (('base', 'datasets_prepared'), ('leakyrandom', 'datasets_leaky_random')):
        with open(os.path.join(ROOT, droot, 'TeaDisease', 'manifest.csv'), newline='') as f:
            rows = {r['file']: r for r in csv.DictReader(f)}
        runs = done_runs(f'teadisease_*_224_{tag}_s*', a.seeds)[:a.limit_runs or None]
        res[tag] = {'runs': {}}
        if not runs:
            continue
        _, root, classes, splits, _, _, _ = load_run(runs[0], 'cpu', random_weights=True)
        items = []
        for rel, c in splits['test']:
            b = boxes_for(rows[rel]) if rel in rows else []
            if b:
                items.append((rel, c, raw_image(root, rel, SIZE), box_mask(b)))
            if a.limit_images and len(items) >= a.limit_images:
                break
        y = np.array([classes.index(i[1]) for i in items])
        res[tag].update({'n_test': len(splits['test']), 'n_with_box': len(items),
                         'box_area_share_by_class': {c: float(np.mean([i[3].mean() for i in items if i[1] == c]))
                                                     for c in classes if any(i[1] == c for i in items)},
                         'n_by_class': {c: int((y == k).sum()) for k, c in enumerate(classes)}})
        for d in runs:
            t0 = time.time()
            try:
                _, _, _, _, model, mean, std = load_run(d, device)
                px = (mean.view(3) * 255).cpu().numpy()
                views = {'original': [i[2] for i in items],
                         'box_only': [np.where(i[3][..., None], i[2], px) for i in items],
                         'box_occluded': [np.where(i[3][..., None], px, i[2]) for i in items]}
                p = {k: predict(model, v, mean, std, device) for k, v in views.items()}
                r = {k: metrics(v, y, len(classes)) for k, v in p.items()}
                if d.endswith('_s0'):               # evidence inside the box (slow: one seed)
                    zero, out = torch.zeros(3).to(device), []
                    for s in range(0, len(items), 16):
                        x = torch.from_numpy(np.stack([i[2] for i in items[s:s + 16]])).permute(0, 3, 1, 2).float().to(device) / 255.0
                        x = (x - mean) / std
                        with torch.no_grad():
                            cls = model(x).float().argmax(1)
                        dr, _ = occlusion(model, x, cls, zero, 8)
                        out.append(focus_scores(dr, [i[3].astype(np.float32) for i in items[s:s + 16]], 8))
                    r['evidence'] = summarise(np.concatenate(out))
                res[tag]['runs'][os.path.basename(d)] = r
                print(f"[BOX] {os.path.basename(d)}: F1 orig={r['original']['macro_f1']:.3f} box-only={r['box_only']['macro_f1']:.3f} "
                      f"box-occluded={r['box_occluded']['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
                del model
            except Exception as e:  # noqa: BLE001
                print(f'[BOX][FAIL] {d}: {type(e).__name__}: {e}', flush=True)
            save_json(res, 'review_boxes', a.suffix)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['cross', 'bg', 'baselines', 'occ', 'box', 'whitespot'])
    p.add_argument('--seeds', type=int, nargs='*', default=None)
    p.add_argument('--limit_runs', type=int, default=0)
    p.add_argument('--limit_images', type=int, default=0)
    p.add_argument('--suffix', default='')
    p.add_argument('--mask', default='current', choices=['current', 'extended'])
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MASK_KIND['name'] = a.mask
    {'cross': task_cross, 'bg': task_bg, 'baselines': task_baselines, 'occ': task_occ, 'box': task_box, 'whitespot': task_whitespot}[a.task](a, device)
    print('done', flush=True)


if __name__ == '__main__':
    main()
