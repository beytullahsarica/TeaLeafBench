"""
Analyses of the field data from saved predictions (no inference, no training). Output: results/_analysis/review_field.json

  sessions    test sessions per class; macro-F1 with and without Black Rot and Healthy; leave-one-session-out macro-F1;
              differences to the reference model with sessions (within class) as the resampling unit
  exposure    random-split runs of TeaDisease: accuracy of test images with a detected near-duplicate in training, with a
              neighbouring training frame of the same class, and with neither
  transfer    per class and per seed: models trained on leaves on paper against models trained on field photographs,
              on the same TeaDisease test images and classes
  sources     exact and leaf-crop duplicates between teaLeafBD and TeaDisease
  variants    results of the sensitivity runs (splits_review)
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from make_review_splits import CUTS, SHARED, frame_no, read_manifest, sessions_of_class  # noqa: E402

RES = os.path.join(ROOT, 'results')
OUT = os.path.join(RES, '_analysis', 'review_field.json')
SHORT_MODELS = ('resnet50', 'efficientnetv2s', 'convnexttiny', 'mobilenetv4m', 'efficientvitb1', 'vitb16', 'swint',
                'maxvitt', 'mambaoutt', 'dinov2s', 'segformerb2', 'yolo11scls')


def macro_f1(y, p, labels):
    fs = []
    for c in labels:
        tp = np.sum((y == c) & (p == c))
        d = np.sum(y == c) + np.sum(p == c)
        fs.append(2 * tp / d if d else 0.0)
    return float(np.mean(fs)) if fs else float('nan')


def read_preds(run, name='test_per_image.csv'):
    with open(os.path.join(run, name), newline='') as f:
        r = list(csv.DictReader(f))
    return [x['path'] for x in r], [x['label'] for x in r], [x['pred'] for x in r]


def runs_of(model, tag='base', ds='teadisease'):
    return [d for d in sorted(glob.glob(os.path.join(RES, f'{ds}_{model}_224_{tag}_s*')))
            if os.path.exists(os.path.join(d, 'test_results.json'))]


def session_ids():
    rows = read_manifest('TeaDisease')
    out, per_class = {}, {}
    for c in sorted({r['class'] for r in rows}):
        sess = sessions_of_class([r for r in rows if r['class'] == c], 12, CUTS.get(c, ()))
        test = 0
        for i, s in enumerate(sess):
            for r in s:
                out[r['file']] = f'{c}/{i}'
            test += any(r['split'] == 'test' for r in s)
        per_class[c] = {'sessions': len(sess), 'test_sessions': test}
    return out, per_class


def sessions_task(models, ref, n_boot, seed=0):
    sid, per_class = session_ids()
    res = {'per_class': per_class, 'models': {}}
    rng = np.random.default_rng(seed)
    data = {}
    for m in models:
        for run in runs_of(m):
            path, y, p = read_preds(run)
            data.setdefault(m, []).append((path, np.array(y), np.array(p)))
    classes = sorted(set(data[next(iter(data))][0][1]))
    without = {'all': classes, 'no_black_rot': [c for c in classes if c != 'black_rot'],
               'no_healthy': [c for c in classes if c != 'healthy'],
               'no_black_rot_no_healthy': [c for c in classes if c not in ('black_rot', 'healthy')]}
    for m, rr in data.items():
        e = {k: [] for k in without}
        loso = []
        per = {c: [] for c in classes}
        for path, y, p in rr:
            for k, cl in without.items():
                keep = np.isin(y, cl)
                e[k].append(macro_f1(y[keep], p[keep], cl))
            for c in classes:
                per[c].append(macro_f1(y, p, [c]))
            s = np.array([sid[x] for x in path])
            vals = []
            for u in sorted(set(s)):            # leave one test session out; a class with no image left is not scored
                keep = s != u
                vals.append(macro_f1(y[keep], p[keep], sorted(set(y[keep]))))
            loso.append((min(vals), max(vals)))
        res['models'][m] = {'n_seeds': len(rr), **{k: [float(np.mean(v)), float(np.std(v))] for k, v in e.items()},
                            'per_class_f1': {c: [float(np.mean(v)), float(np.std(v))] for c, v in per.items()},
                            'leave_one_session_out_min_max': [float(np.mean([a for a, _ in loso])),
                                                              float(np.mean([b for _, b in loso]))]}
    # differences to the reference model: sessions within a class are resampled, seeds are resampled; a class with one
    # test session cannot vary at this level, so the interval is conditional on that session
    refr = data[ref]
    sess = np.array([sid[x] for x in refr[0][0]])
    order = {x: i for i, x in enumerate(refr[0][0])}
    by_class = {}
    for i, x in enumerate(refr[0][0]):
        by_class.setdefault(refr[0][1][i], {}).setdefault(sess[i], []).append(i)
    res['classes_with_one_test_session'] = [c for c, v in by_class.items() if len(v) == 1]

    def align(rr):
        out = []
        for path, y, p in rr:
            idx = np.array([order[x] for x in path])
            yy, pp = np.empty(len(y), dtype=y.dtype), np.empty(len(p), dtype=object)
            yy[idx], pp[idx] = y, p
            out.append((yy, pp.astype(str)))
        return out
    aligned = {m: align(rr) for m, rr in data.items()}
    draws = []
    for _ in range(n_boot):
        ids = []
        for c, s in by_class.items():
            keys = sorted(s)
            for k in rng.integers(0, len(keys), len(keys)):
                ids.extend(s[keys[int(k)]])
        draws.append(np.array(ids))
    res['difference_to_reference'] = {'reference': ref}
    for m in aligned:
        d = []
        for ids in draws:
            a = aligned[m][int(rng.integers(len(aligned[m])))]
            b = aligned[ref][int(rng.integers(len(aligned[ref])))]
            d.append(macro_f1(a[0][ids], a[1][ids], classes) - macro_f1(b[0][ids], b[1][ids], classes))
        obs = np.mean([macro_f1(a[0], a[1], classes) for a in aligned[m]]) - \
            np.mean([macro_f1(b[0], b[1], classes) for b in aligned[ref]])
        res['difference_to_reference'][m] = {'mean': float(obs), 'ci95': [float(np.percentile(d, 2.5)),
                                                                          float(np.percentile(d, 97.5))]}
    return res


def exposure_task(models):
    with open(os.path.join(ROOT, 'datasets_leaky_random', 'TeaDisease', 'manifest.csv'), newline='') as f:
        rows = [r for r in csv.DictReader(f)]
    train = [r for r in rows if r['split'] == 'train']
    groups = {r['dup_group'] for r in train if r['dup_group'] not in ('', '-1')}
    near = {}
    for r in train:
        near.setdefault(r['class'], set()).add(frame_no(r['file']))
    kind = {}
    for r in rows:
        if r['split'] != 'test':
            continue
        n = frame_no(r['file'])
        if r['dup_group'] not in ('', '-1') and r['dup_group'] in groups:
            kind[r['file']] = 'detected_duplicate'
        elif any(abs(n - x) <= 2 for x in near.get(r['class'], ())):
            kind[r['file']] = 'neighbour_frame_only'
        else:
            kind[r['file']] = 'neither'
    counts = {k: sum(v == k for v in kind.values()) for k in ('detected_duplicate', 'neighbour_frame_only', 'neither')}
    res = {'n_test': len(kind), 'counts': counts, 'models': {}}
    for m in models:
        acc = {k: [] for k in counts}
        for run in runs_of(m, 'leakyrandom'):
            path, y, p = read_preds(run)
            for k in counts:
                c = [yy == pp for x, yy, pp in zip(path, y, p) if kind.get(x) == k]
                if c:
                    acc[k].append(float(np.mean(c)))
        res['models'][m] = {k: [float(np.mean(v)), float(np.std(v))] if v else None for k, v in acc.items()}
    return res


def restricted(run, keep_names):
    z = np.load(os.path.join(run, 'test_probs.npz'))
    classes = [str(c) for c in z['classes']]
    keep = [classes.index(c) for c in keep_names]
    rows = np.isin(z['labels'], keep)
    y = np.array([keep.index(int(v)) for v in z['labels'][rows]])
    return y, z['probs'][rows][:, keep].argmax(1)


def transfer_task(models, tags=('base',)):
    names = sorted(SHARED)                                   # TeaDisease class names; index = same order on both sides
    res = {'classes': names, 'models': {}}
    for m in models:
        e = {'field_trained': [], 'by_class': {'field_trained': {c: [] for c in names}, 'paper_trained': {c: [] for c in names}}}
        for run in runs_of(m):
            y, p = restricted(run, names)
            e['field_trained'].append(macro_f1(y, p, range(3)))
            for i, c in enumerate(names):
                e['by_class']['field_trained'][c].append(macro_f1(y, p, [i]))
        for tag in tags:
            key = 'paper_trained' if tag == 'base' else f'paper_trained_{tag}'
            e[key] = []
            for run in runs_of(m, tag, 'tealeafbd'):
                f = os.path.join(run, 'test_per_image_on-teadisease.csv')
                if not os.path.exists(f):
                    continue
                _, yy, pp = read_preds(run, 'test_per_image_on-teadisease.csv')
                back = {v: k for k, v in SHARED.items()}
                y = np.array([names.index(back[x]) for x in yy])
                p = np.array([names.index(back[x]) for x in pp])
                e[key].append(macro_f1(y, p, range(3)))
                if tag == 'base':
                    for i, c in enumerate(names):
                        e['by_class']['paper_trained'][c].append(macro_f1(y, p, [i]))
        res['models'][m] = e
    return res


def sources_task(limit=0, thr=6.0):
    from prepare_data import features  # noqa: E402
    a = [r for r in read_manifest('TeaLeafBD')]
    b = [r for r in read_manifest('TeaDisease')]
    if limit:
        a, b = a[:limit], b[:limit]
    fa = [features(os.path.join(ROOT, 'datasets_prepared', 'TeaLeafBD', r['file'])) for r in a]
    fb = [features(os.path.join(ROOT, 'datasets_prepared', 'TeaDisease', r['file'])) for r in b]
    fa, fb = [x for x in fa if not isinstance(x, str)], [x for x in fb if not isinstance(x, str)]
    md5 = len({x[0] for x in fa} & {x[0] for x in fb})
    ta = np.stack([x[2] for x in fa]).astype(np.float32)
    tb = np.stack([x[2] for x in fb]).astype(np.float32)
    best = np.full(len(tb), np.inf)
    for i in range(len(tb)):
        best[i] = np.abs(ta - tb[i]).mean(1).min()
    return {'n_teaLeafBD': len(fa), 'n_TeaDisease': len(fb), 'identical_md5': md5, 'leaf_crop_pairs_within_threshold':
            int((best <= thr).sum()), 'threshold': thr, 'smallest_difference': float(best.min()),
            'difference_percentiles_1_50': [float(np.percentile(best, 1)), float(np.percentile(best, 50))]}


def capped_task(models=('convnexttiny', 'efficientvitb1', 'swint', 'maxvitt'), n_boot=2000, seed=0):
    """Both sides capped per scored class (tags cap and eq), same test images and classes: paper-trained minus field-trained,
    test images within a class and seeds resampled. All three classes have one test session, so sessions cannot be resampled."""
    names = sorted(SHARED)
    back = {v: k for k, v in SHARED.items()}
    rng = np.random.default_rng(seed)
    data = {}
    for m in models:
        field, paper = [], []
        for run in runs_of(m, 'cap'):
            path, label, _ = read_preds(run)
            z = np.load(os.path.join(run, 'test_probs.npz'))
            classes = [str(c) for c in z['classes']]
            keep = [classes.index(c) for c in names]
            rows = np.isin(z['labels'], keep)
            assert all(classes[int(v)] == x for v, x in zip(z['labels'], label))
            pred = z['probs'][rows][:, keep].argmax(1)
            field.append({x: int(p) for x, p, r in zip(np.array(path)[rows], pred, [1] * len(pred)) if r})
            truth = {x: names.index(c) for x, c in zip(np.array(path)[rows], np.array(label)[rows])}
        for run in runs_of(m, 'eq', 'tealeafbd'):
            path, _, pp = read_preds(run, 'test_per_image_on-teadisease.csv')
            paper.append({x: names.index(back[p]) for x, p in zip(path, pp)})
        data[m] = (field, paper)
    ids = sorted(truth)
    y = np.array([truth[x] for x in ids])
    by = {m: tuple([np.array([run[x] for x in ids]) for run in side] for side in data[m]) for m in models}
    res = {'classes': names, 'n_test': len(ids), 'models': {}}
    draws = {m: [] for m in models}
    for _ in range(n_boot):
        s = np.concatenate([rng.choice(np.nonzero(y == c)[0], int((y == c).sum())) for c in range(len(names))])
        for m in models:
            f = by[m][0][int(rng.integers(len(by[m][0])))]
            p = by[m][1][int(rng.integers(len(by[m][1])))]
            draws[m].append(macro_f1(y[s], p[s], range(3)) - macro_f1(y[s], f[s], range(3)))
    for m in models:
        fs = [macro_f1(y, f, range(3)) for f in by[m][0]]
        ps = [macro_f1(y, p, range(3)) for p in by[m][1]]
        res['models'][m] = {'field_capped': fs, 'paper_capped': ps, 'difference': float(np.mean(ps) - np.mean(fs)),
                            'ci95': [float(x) for x in np.percentile(draws[m], [2.5, 97.5])],
                            'by_class': {'field_capped': {c: float(np.mean([macro_f1(y, f, [i]) for f in by[m][0]])) for i, c in enumerate(names)},
                                         'paper_capped': {c: float(np.mean([macro_f1(y, p, [i]) for p in by[m][1]])) for i, c in enumerate(names)}}}
    mean = np.mean([draws[m] for m in models], axis=0)
    res['mean_difference'] = float(np.mean([res['models'][m]['difference'] for m in models]))
    res['mean_ci95'] = [float(x) for x in np.percentile(mean, [2.5, 97.5])]
    return res


def variants_task(models):
    res = {}
    names = sorted(SHARED)
    for tag in ('grouped', 'p1', 'p2', 'p3', 'gap30', 'f25', 'f50'):
        for m in models:
            vals, shared, acc = [], [], []
            for run in runs_of(m, tag):
                with open(os.path.join(run, 'test_results.json')) as f:
                    r = json.load(f)
                vals.append(r['macro_f1'])
                acc.append(r.get('accuracy', float('nan')))
                y, p = restricted(run, names)
                shared.append(macro_f1(y, p, range(3)))
            if vals:
                res.setdefault(tag, {})[m] = {'macro_f1': [float(np.mean(vals)), float(np.std(vals))],
                                              'accuracy': [float(np.mean(acc)), float(np.std(acc))],
                                              'shared_classes_f1': [float(np.mean(shared)), float(np.std(shared))],
                                              'n': len(vals)}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='*', default=['sessions', 'exposure', 'transfer', 'sources', 'variants'])
    p.add_argument('--ref', default='vitb16')
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--limit_models', type=int, default=0)
    p.add_argument('--limit_images', type=int, default=0)
    p.add_argument('--out', default=OUT)
    a = p.parse_args()
    models = list(SHORT_MODELS[:a.limit_models] if a.limit_models else SHORT_MODELS)
    if a.ref not in models:
        models.append(a.ref)
    res = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            res = json.load(f)
    for t in a.tasks:
        print('===', t, flush=True)
        if t == 'sessions':
            res[t] = sessions_task(models, a.ref, a.n_boot)
        elif t == 'exposure':
            res[t] = exposure_task(models)
        elif t == 'transfer':
            res[t] = transfer_task(models, ('base', 'eq', 'ratio'))
        elif t == 'sources':
            res[t] = sources_task(a.limit_images)
        elif t == 'variants':
            res[t] = variants_task(models)
        elif t == 'capped':
            res[t] = capped_task()
        new = res[t]
        if os.path.exists(a.out):            # another job may have written other tasks in the meantime: keep them
            with open(a.out) as f:
                res = json.load(f)
        res[t] = new
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res[t])[:1500], flush=True)


if __name__ == '__main__':
    main()
