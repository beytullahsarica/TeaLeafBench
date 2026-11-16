"""
Collect all finished runs (results/*/test_results*.json + config.json + complexity.json) into paper tables.

  python tools/make_paper_tables.py                     # plain test results
  python tools/make_paper_tables.py --suffix _tta-flip  # results with flip TTA
  python tools/make_paper_tables.py --tag base --ref convnext_tiny

Outputs (paper/tables/):
  summary.csv                         every (dataset, model, img_size, tag) with mean / std over seeds
  main_<dataset>.tex                  benchmark table per dataset (mean ± std over seeds, best in bold)
  efficiency.tex                      params / GFLOPs / FPS / latency per model
  mcnemar_<dataset>.tex, mcnemar.tex  McNemar test of every model against --ref (default: best mean VALIDATION
                                      macro-F1), per seed on the per-image predictions of the same seed, Holm per seed
  paired_ci.tex                       macro-F1 / accuracy difference to the reference, paired bootstrap over test
                                      images and seeds (95 % interval)
  calibration.tex                     ECE (15 equal-width bins), NLL, Brier; models and the top-3 ensemble
  leakage.tex                         leakage-free split vs leaky splits (tags leaky-random / leaky-official), seeds
  leakage_decomp.tex                  TeaDisease random - session: Acc / MCC / F1_m / Bal-Acc / F1_m w/o Black Rot,
                                      and accuracy on the test images both splits share
  leakage_perclass.tex                TeaDisease per-class F1 on the three splits (all models and seeds)
  cross.tex                           own test set vs other dataset, both on the shared classes (restricted and all
                                      outputs), mean over the seeds that were tested
  leaf_focus.tex                      leaf share of Grad-CAM / Grad-CAM++ / occlusion energy (uniform background) and
                                      the ceiling-aware occlusion index (tools/background_test.py)
  background.tex                      background interventions and the border-colour classifier (background_test.py)
  ensembles.tex                       per seed: soft voting of the top-3 and of the top model of 3 model categories (validation)
main / McNemar / efficiency use the base runs only (--tag to choose another tag); summary.csv has every tag.
Missing values print as "--" so the paper always compiles.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.registry import BASELINES, DEFAULT_BENCHMARK, display_name, family, normalize_name  # noqa: E402

METRICS = [('accuracy', 'Acc'), ('balanced_accuracy', 'Bal-Acc'), ('macro_precision', 'Pr$_m$'),
           ('macro_recall', 'Re$_m$'), ('macro_f1', 'F1$_m$'), ('kappa', '$\\kappa$'), ('mcc', 'MCC')]
FAMILY_ORDER = ['cnn', 'light', 'transformer', 'hybrid', 'mamba', 'foundation', 'segmentation', 'yolo', 'other',
                'proposed']
FAMILY_NAMES = {'cnn': 'CNNs', 'light': 'Lightweight', 'transformer': 'Transformers', 'hybrid': 'Hybrid CNN--Transformer',
                'mamba': 'Mamba-style', 'foundation': 'Foundation model', 'segmentation': 'Segmentation encoder',
                'yolo': 'YOLO', 'other': 'Other', 'proposed': 'Proposed'}


def load_runs(results_dir, suffix):
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(results_dir, '*', f'test_results{suffix}.json'))):
        d = os.path.dirname(p)
        if os.path.basename(d).startswith('smoke') or not os.path.exists(os.path.join(d, 'config.json')):
            continue                              # smoke tests (pipeline checks) are never part of the tables
        with open(os.path.join(d, 'config.json')) as f:
            args = json.load(f)['args']
        with open(p) as f:
            r = json.load(f)
        if r.get('test_dataset', args['dataset']) != args['dataset']:
            continue                              # cross-dataset runs are not part of the main tables
        cx = os.path.join(d, 'complexity.json')
        if os.path.exists(cx):
            with open(cx) as f:
                r.update({f'cx_{k}': v for k, v in json.load(f).items()})
        r['_dir'], r['_seed'] = d, args['seed']
        groups[(args['dataset'], normalize_name(args['model']), args['img_size'], args.get('tag', 'base'))].append(r)
    return groups


def stat(runs, key):
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0), len(vals)


def fmt(s, scale=100.0, digits=2, bold=False):
    if s is None:
        return '--'
    m, sd, n = s
    txt = f'{m * scale:.{digits}f}'
    if bold:
        txt = f'\\textbf{{{txt}}}'
    if n > 1:
        txt += f'{{\\scriptsize$\\pm${sd * scale:.{digits}f}}}'
    return txt


def mcnemar(a_correct, b_correct):
    """Exact (binomial) McNemar test on paired correctness vectors -> (n01, n10, p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as the input)."""
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def ptxt(p):
    return f'{p:.3g}' if p >= 1e-4 else '$<$1e-4'


def write_mcnemar_combined(mc, out_dir, suffix):
    """mcnemar.tex: one column block per dataset against that dataset's reference (best mean validation macro-F1):
    n01, n10 and Holm-adjusted p of seed 0, and in how many seeds the difference was significant (Holm, 0.05)."""
    dss = [ds for ds in ('TeaLeafBD', 'TeaSickness', 'TeaDisease') if ds in mc]
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% exact McNemar test vs the reference (best mean validation macro-F1), per seed, p Holm-adjusted per dataset '
         'and seed; n01 / n10 / p_Holm of seed 0; Sig.: seeds with p_Holm < 0.05',
         '\\begin{tabular}{l' + 'rrrc' * len(dss) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{DS_LABEL[ds]} (ref.\\ {display_name(mc[ds][0])})}}' for ds in dss) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(len(dss))),
         'Model & ' + ' & '.join(['$n_{01}$ & $n_{10}$ & $p_\\mathrm{Holm}$ & Sig.'] * len(dss)) + ' \\\\', '\\midrule']
    for model in DEFAULT_BENCHMARK:
        cells = []
        for ds in dss:
            ref, tests = mc[ds]
            t = next((x for x in tests if x[0] == model), None)
            cells += (['\\multicolumn{4}{c}{reference}'] if model == ref else
                      [str(t[1]), str(t[2]), ptxt(t[4]), f'{t[5]}/{t[6]}'] if t else ['--'] * 4)
        L.append(f'{display_name(model)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, f'mcnemar{suffix}.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"mcnemar{suffix}.tex: {', '.join(dss)}")


def load_probs(run_dir, suffix=''):
    """(labels, probs, classes) of a run's test set, or None. The test images are in the same order in every run
    of a dataset and split (sorted split file)."""
    path = os.path.join(run_dir, f'test_probs{suffix}.npz')
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z['labels'].astype(int), z['probs'].astype(np.float64), [str(c) for c in z['classes']]


def macro_f1_fast(y, p, k, classes=None):
    """Macro-F1 over the classes present in y (or the given class indices), from the confusion counts."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    keep = np.nonzero(cm.sum(1) > 0)[0] if classes is None else np.asarray(classes)
    return float(f1[keep].mean())


def boot_diff(y, preds_a, preds_b, k, rng, n_boot=2000):
    """Paired bootstrap of the difference a - b in macro-F1 and accuracy (mean over the seeds of each model): the
    test images are resampled (the same images for both models) and the seeds of each model are resampled, so the
    interval includes both test-sample and training (seed) variation. -> (dF1, lo, hi), (dAcc, lo, hi)"""
    n = len(y)
    pa, pb = np.stack(preds_a), np.stack(preds_b)
    est_f1 = np.mean([macro_f1_fast(y, p, k) for p in pa]) - np.mean([macro_f1_fast(y, p, k) for p in pb])
    est_acc = (pa == y).mean() - (pb == y).mean()
    bf, ba = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        fa = np.array([macro_f1_fast(yb, p[idx], k) for p in pa])
        fb = np.array([macro_f1_fast(yb, p[idx], k) for p in pb])
        ca, cb = (pa[:, idx] == yb).mean(1), (pb[:, idx] == yb).mean(1)
        sa, sb = rng.integers(0, len(pa), len(pa)), rng.integers(0, len(pb), len(pb))
        bf[b] = fa[sa].mean() - fb[sb].mean()
        ba[b] = ca[sa].mean() - cb[sb].mean()
    return ((est_f1, *np.percentile(bf, [2.5, 97.5])), (est_acc, *np.percentile(ba, [2.5, 97.5])))


def brier(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def mean_val_f1(runs):
    v = [val_macro_f1(r['_dir']) for r in runs]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else -1.0


def read_correct(run_dir, suffix):
    path = os.path.join(run_dir, f'test_per_image{suffix}.csv')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r['path']: r['correct'] == '1' for r in rows}


def sort_key(item):
    (ds, model, size, tag), _ = item
    fam = family(model)
    order = (DEFAULT_BENCHMARK.index(model) if model in DEFAULT_BENCHMARK else
             100 + list(BASELINES).index(model) if model in BASELINES else 999)
    return ds, FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 9, order, model, size, tag


def pm1(runs, key):
    """mean +- sd over seeds (%), one decimal; '--' without runs"""
    s = stat(runs, key)
    return fmt(s, digits=1) if s else '--'


def paired_delta(base, leaky, fn):
    """fn(run) of the leaky minus the leakage-free run of the same seed, for every seed both have -> list"""
    b = {r['_seed']: r for r in base}
    out = []
    for r in leaky:
        if r['_seed'] in b:
            x, y = fn(r), fn(b[r['_seed']])
            if x is not None and y is not None:
                out.append(x - y)
    return out


def dtxt(vals, scale=100.0):
    if not vals:
        return '--'
    m = np.mean(vals) * scale
    t = f'{m:+.1f}'.replace('-', '$-$')
    return t + (f'{{\\scriptsize$\\pm${np.std(vals, ddof=1) * scale:.1f}}}' if len(vals) > 1 else '')


def write_leakage(groups, out_dir):
    """leakage.tex: runs of the same model and recipe on the leakage-free split and on the leaky splits (TeaDisease:
    random image-level and the released detection split; other datasets: random image-level), mean +- sd over seeds."""
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% Acc / F1_m in %, mean +- sd over seeds; each split has its own test set (same images overall, White Spot '
         'excluded)', '\\begin{tabular}{l' + 'cc' * 3 + '}', '\\toprule',
         ' & \\multicolumn{2}{c}{Grouped split} & \\multicolumn{2}{c}{Random split} & '
         '\\multicolumn{2}{c}{Released split} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
         'Model & Acc & F1$_m$ & Acc & F1$_m$ & Acc & F1$_m$ \\\\', '\\midrule',
         '\\multicolumn{7}{l}{\\emph{TeaDisease} (grouped: session split)} \\\\']
    n = 0
    for model in DEFAULT_BENCHMARK:
        rs = [groups.get(('TeaDisease', model, 224, t), []) for t in ('base', 'leaky-random', 'leaky-official')]
        if not any(rs):
            continue
        n += 1
        L.append(f'{display_name(model)} & ' + ' & '.join(f'{pm1(r, "accuracy")} & {pm1(r, "macro_f1")}' for r in rs)
                 + ' \\\\')
    for ds in ('TeaLeafBD', 'TeaSickness'):
        rows = []
        for model in DEFAULT_BENCHMARK:
            clean, rand = groups.get((ds, model, 224, 'base'), []), groups.get((ds, model, 224, 'leaky-random'), [])
            if rand:
                rows.append(f'{display_name(model)} & {pm1(clean, "accuracy")} & {pm1(clean, "macro_f1")} & '
                            f'{pm1(rand, "accuracy")} & {pm1(rand, "macro_f1")} & -- & -- \\\\')
        if rows:
            n += len(rows)
            L += ['\\midrule', f'\\multicolumn{{7}}{{l}}{{\\emph{{{DS_LABEL[ds]}}} (grouped: duplicate-grouped '
                               f'split)}} \\\\'] + rows
    L += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, 'leakage.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"leakage.tex: {n} rows")


def f1_without(run, cls_name):
    """macro-F1 of a run's test predictions over all classes except cls_name (the predictions are not changed)"""
    z = load_probs(run['_dir'])
    if z is None or cls_name not in z[2]:
        return None
    y, p, classes = z[0], z[1].argmax(1), z[2]
    keep = [i for i, c in enumerate(classes) if c != cls_name and (y == i).any()]
    return macro_f1_fast(y, p, len(classes), keep)


def acc_on(run, names):
    """accuracy of a run on the test images with these file names (basename), None if it has none of them"""
    path = os.path.join(run['_dir'], 'test_per_image.csv')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        c = [r['correct'] == '1' for r in csv.DictReader(f) if os.path.basename(r['path']) in names]
    return float(np.mean(c)) if c else None


def test_names(runs):
    path = os.path.join(runs[0]['_dir'], 'test_per_image.csv') if runs else ''
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {os.path.basename(r['path']) for r in csv.DictReader(f)}


def write_leakage_decomp(groups, out_dir, ds='TeaDisease', drop='black_rot'):
    """leakage_decomp.tex: TeaDisease, random image-level split minus session split for the same model and seed, in
    several metrics: accuracy, MCC, macro-F1, balanced accuracy (independent of the class proportions of the test
    set), macro-F1 without Black Rot, and the accuracy of both models on the test images the two splits share
    (identical images, so the test sample is not a factor). Mean +- sd over the paired seeds."""
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% random split minus session split, same model and seed (percentage points, mean +- sd over seeds); '
         'shared: test images that are in both test sets',
         '\\begin{tabular}{lccccccc}', '\\toprule',
         ' & \\multicolumn{5}{c}{Own test sets: $\\Delta$ (random $-$ session)} & '
         '\\multicolumn{2}{c}{Shared test images: Acc} \\\\', '\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}',
         'Model & Acc & MCC & F1$_m$ & Bal-Acc & F1$_m$ w/o Black Rot & session-trained & random-trained \\\\',
         '\\midrule']
    allv = defaultdict(list)
    shared, n_shared = None, 0
    for model in DEFAULT_BENCHMARK:
        base, rand = groups.get((ds, model, 224, 'base'), []), groups.get((ds, model, 224, 'leaky-random'), [])
        if not base or not rand:
            continue
        if shared is None:
            shared = test_names(base) & test_names(rand)
            n_shared = len(shared)
        cols = [paired_delta(base, rand, lambda r, k=k: r.get(k)) for k in ('accuracy', 'mcc', 'macro_f1',
                                                                          'balanced_accuracy')]
        cols.append(paired_delta(base, rand, lambda r: f1_without(r, drop)))
        sb = [x for x in (acc_on(r, shared) for r in base) if x is not None]
        sr = [x for x in (acc_on(r, shared) for r in rand) if x is not None]
        for i, v in enumerate(cols):
            allv[i] += [np.mean(v)] if v else []
        allv['sb'] += [np.mean(sb)] if sb else []
        allv['sr'] += [np.mean(sr)] if sr else []
        L.append(f'{display_name(model)} & ' + ' & '.join(dtxt(v) for v in cols) + ' & ' +
                 ' & '.join(fmt((np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0, len(v)), digits=1) if v else '--'
                            for v in (sb, sr)) + ' \\\\')
    if allv:
        L += ['\\midrule', 'Mean of the models & ' + ' & '.join(f'{np.mean(allv[i]) * 100:+.1f}'.replace('-', '$-$')
                                                                for i in range(5)) +
              f" & {np.mean(allv['sb']) * 100:.1f} & {np.mean(allv['sr']) * 100:.1f} \\\\"]
    L += ['\\bottomrule', '\\end{tabular}', f'% shared test images: {n_shared}']
    with open(os.path.join(out_dir, 'leakage_decomp.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"leakage_decomp.tex: {n_shared} shared test images")


def write_leakage_perclass(groups, out_dir, ds='TeaDisease'):
    """leakage_perclass.tex: per-class F1 on the three TeaDisease splits, mean +- sd over all models and seeds,
    with the number of test images of each class in each split."""
    from data.class_names import display_name as class_display
    tags = ('base', 'leaky-random', 'leaky-official')
    f1s, support = defaultdict(list), {}
    for (d, model, size, tag), runs in groups.items():
        if d != ds or tag not in tags or size != 224 or model not in DEFAULT_BENCHMARK:
            continue
        for r in runs:
            pc = r.get('per_class') or {}
            for c, f, n in zip(pc.get('class', []), pc.get('f1', []), pc.get('support', [])):
                f1s[(tag, c)].append(f)
                support[(tag, c)] = n
    classes = sorted({c for _, c in f1s})
    if not classes:
        return
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% per-class F1 (%), mean +- sd over all models and seeds; n: test images of the class in that split',
         '\\begin{tabular}{lcccccc}', '\\toprule',
         ' & \\multicolumn{2}{c}{Session split} & \\multicolumn{2}{c}{Random split} & '
         '\\multicolumn{2}{c}{Released split} \\\\', '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
         'Class & F1 & $n$ & F1 & $n$ & F1 & $n$ \\\\', '\\midrule']
    for c in classes:
        cells = []
        for t in tags:
            v = f1s.get((t, c), [])
            cells += [fmt((np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0, len(v)), digits=1) if v else '--',
                      str(support.get((t, c), '--'))]
        L.append(f'{class_display(c)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, 'leakage_perclass.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"leakage_perclass.tex: {len(classes)} classes")


CROSS = [('TeaLeafBD', 'TeaSickness'), ('TeaSickness', 'TeaLeafBD'), ('TeaLeafBD', 'TeaDisease'),
         ('TeaDisease', 'TeaLeafBD')]
# dataset labels printed in tables and figures (folder names stay TeaLeafBD / TeaSickness / TeaDisease)
DS_LABEL = {'TeaLeafBD': 'teaLeafBD', 'TeaSickness': 'Tea Sickness', 'TeaDisease': 'TeaDisease'}
SHORT = DS_LABEL


def shared_class_f1(run_dir, shared):
    """Macro-F1 of a run on its OWN test set restricted to the given classes (rows and outputs), i.e. the
    in-dataset reference for a cross-dataset test with the same classes."""
    from sklearn.metrics import f1_score
    path = os.path.join(run_dir, 'test_probs.npz')
    if not os.path.exists(path):
        return None
    z = np.load(path)
    classes = [str(c) for c in z['classes']]
    keep = [classes.index(c) for c in shared]
    rows = np.isin(z['labels'], keep)
    labels = np.array([keep.index(int(v)) for v in z['labels'][rows]])
    pred = z['probs'][rows][:, keep].argmax(1)
    return float(f1_score(labels, pred, labels=list(range(len(keep))), average='macro', zero_division=0))


def unrestricted_f1(run_dir, test_dataset, shared):
    """Cross-dataset macro-F1 over the shared classes with ALL outputs of the model: a prediction of a class that
    the test dataset does not have counts as an error (test_probs_on-<ds>_allclasses.npz, written by train.py)."""
    path = os.path.join(run_dir, f'test_probs_on-{test_dataset.lower()}_allclasses.npz')
    if not os.path.exists(path):
        return None
    z = np.load(path)
    classes = [str(c) for c in z['classes']]
    return macro_f1_fast(z['labels'].astype(int), z['probs'].argmax(1), len(classes),
                         [classes.index(c) for c in shared])


def write_cross(results_dir, out_dir):
    """cross.tex: base models tested on another dataset (shared classes only), next to the same model on its own
    test set restricted to those classes, mean +- sd over the seeds that were tested; Cross: outputs restricted to
    the shared classes, All: all outputs (predictions of other classes are errors)."""
    res = defaultdict(list)
    for p in glob.glob(os.path.join(results_dir, '*_base_s*', 'test_results_on-*.json')):
        with open(p) as f:
            r = json.load(f)
        if r.get('tag', 'base') != 'base' or r.get('tta', 'none') != 'none':
            continue
        res[(r['dataset'], r['test_dataset'], normalize_name(r['model']))].append((r, os.path.dirname(p)))
    if not res:
        return

    def cell(vals):
        vals = [v for v in vals if v is not None]
        return fmt((np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0, len(vals)), digits=1) if vals else '--'

    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% F1_m (%) over the shared classes, mean +- sd over seeds: In = own test set, Cross = other dataset\'s test '
         'set with outputs restricted to the shared classes, All = all outputs',
         '\\begin{tabular}{l' + 'ccc' * len(CROSS) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{SHORT[s]}$\\rightarrow${SHORT[t]}}}' for s, t in CROSS) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(CROSS))),
         'Model & ' + ' & '.join(['In & Cross & All'] * len(CROSS)) + ' \\\\', '\\midrule']
    for model in DEFAULT_BENCHMARK:
        cells, any_ = [], False
        for s, t in CROSS:
            hits = res.get((s, t, model), [])
            if not hits:
                cells += ['--'] * 3
                continue
            any_ = True
            shared = hits[0][0]['per_class']['class']
            cells += [cell([shared_class_f1(d, shared) for _, d in hits]), cell([r['macro_f1'] for r, _ in hits]),
                      cell([unrestricted_f1(d, t, shared) for _, d in hits])]
        if any_:
            L.append(f'{display_name(model)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    for (s, t) in CROSS:
        hit = next((v[0][0] for k, v in res.items() if k[:2] == (s, t)), None)
        if hit:
            pc = hit['per_class']
            L.append(f"% {s} -> {t}: {len(pc['class'])} shared classes {pc['class']}, support {pc['support']}, "
                     f"{hit['n']} test images")
    with open(os.path.join(out_dir, 'cross.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"cross.tex: {sum(len(v) for v in res.values())} cross-dataset tests")


def background_results(main_items):
    """(dataset, model) -> list of background_test.json of the base runs (tools/background_test.py)"""
    out = defaultdict(list)
    for (ds, model, size, tag), runs in main_items:
        for r in runs:
            p = os.path.join(r['_dir'], 'background_test.json')
            if size == 224 and os.path.exists(p):
                with open(p) as f:
                    out[(ds, model)].append(json.load(f))
    return out


def write_leaf_focus(main_items, out_dir):
    """leaf_focus.tex: share of the saliency energy on the leaf (images with a uniform background), mean over
    the seeds of the base runs, for Grad-CAM, Grad-CAM++ and occlusion, and the ceiling-aware occlusion index
    (S - A) / (S_max - A) of tools/background_test.py (0: no preference for the leaf, 1: all evidence on the leaf)."""
    keys = [('leaf_energy_share', 'Grad-CAM'), ('pp_leaf_energy_share', 'Grad-CAM++'),
            ('occ_leaf_energy_share', 'Occl.')]
    dss = [ds for ds in ('TeaLeafBD', 'TeaSickness') if any(k[0] == ds for k, _ in main_items)]
    by = {(k[0], k[1]): v for k, v in main_items if k[2] == 224}
    bg = background_results(main_items)
    nk = len(keys) + 1
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% leaf share (%) of the saliency energy, test images with a uniform background; Index: ceiling-aware '
         'occlusion index (S - A) / (S_max - A), x100; mean over seeds',
         '\\begin{tabular}{l' + 'c' * (nk * len(dss)) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{{nk}}}{{c}}{{{DS_LABEL[ds]}}}' for ds in dss) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + nk * i}-{1 + nk * (i + 1)}}}' for i in range(len(dss))),
         'Model & ' + ' & '.join(h for _ in dss for h in [h for _, h in keys] + ['Index']) + ' \\\\', '\\midrule']
    n = 0
    for model in DEFAULT_BENCHMARK:
        cells = []
        for ds in dss:
            cells += [fmt(stat(by.get((ds, model), []), k), digits=1) for k, _ in keys[:-1]]
            # occlusion share from background_test.json: same images as the index; images without any positive drop
            # are skipped there, while one such image makes the mean stored at training time NaN
            occ = [b['occlusion']['leaf_share'] for b in bg.get((ds, model), [])]
            cells.append(fmt((np.mean(occ), np.std(occ, ddof=1) if len(occ) > 1 else 0, len(occ)), digits=1)
                         if occ else '--')
            idx = [b['occlusion']['focus_index'] for b in bg.get((ds, model), [])]
            cells.append(fmt((np.mean(idx), np.std(idx, ddof=1) if len(idx) > 1 else 0, len(idx)), digits=1)
                         if idx else '--')
        if any(c != '--' for c in cells):
            n += 1
            L.append(f'{display_name(model)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    for ds in dss:
        a = [b['occlusion']['area_share'] for v in (bg.get((ds, m), []) for m in DEFAULT_BENCHMARK) for b in v]
        s = [b['occlusion']['max_cell_share'] for v in (bg.get((ds, m), []) for m in DEFAULT_BENCHMARK) for b in v]
        if a:
            L.append(f'% {ds}: mean leaf area share {np.mean(a):.3f}, mean largest 8x8-cell leaf share {np.mean(s):.3f}')
    with open(os.path.join(out_dir, 'leaf_focus.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"leaf_focus.tex: {n} models")


def write_background(main_items, results_dir, out_dir):
    """background.tex: accuracy (%) on the test images with a uniform background, original and with the background
    or the leaf removed (tools/background_test.py), mean over seeds; Follows: share of predictions that follow a
    background colour taken from another class. Last rows: a classifier on the border colour alone and the
    majority class."""
    bg = background_results(main_items)
    dss = [ds for ds in ('TeaLeafBD', 'TeaSickness') if any(k[0] == ds for k in bg)]
    if not dss:
        return
    cols = [('original', 'accuracy', 'Original'), ('leaf_only', 'accuracy', 'Leaf only'),
            ('background_only', 'accuracy', 'Backgr.\\ only'), ('swapped', 'accuracy', 'Swapped'),
            ('swapped', 'follows_background', 'Follows')]
    nc = len(cols)
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% accuracy (%) on the test images with a uniform background, mean +- sd over seeds; Follows: share (%) of '
         'predictions equal to the class whose background colour was inserted',
         '\\begin{tabular}{l' + 'c' * (nc * len(dss)) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{{nc}}}{{c}}{{{DS_LABEL[ds]}}}' for ds in dss) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + nc * i}-{1 + nc * (i + 1)}}}' for i in range(len(dss))),
         'Model & ' + ' & '.join(h for _ in dss for _, _, h in cols) + ' \\\\', '\\midrule']
    for model in DEFAULT_BENCHMARK:
        cells = []
        for ds in dss:
            rs = bg.get((ds, model), [])
            for cond, key, _ in cols:
                v = [r['conditions'][cond][key] for r in rs]
                cells.append(fmt((np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0, len(v)), digits=1) if v else '--')
        if any(c != '--' for c in cells):
            L.append(f'{display_name(model)} & ' + ' & '.join(cells) + ' \\\\')
    p = os.path.join(results_dir, '_analysis', 'border_colour.json')
    if os.path.exists(p):
        with open(p) as f:
            bc = json.load(f)
        L.append('\\midrule')
        for key, label in (('border_only', 'Border colour only (logistic regression)'),
                           ('majority_class', 'Majority class')):
            cells = []
            for ds in dss:
                v = bc.get(ds, {}).get(key)
                cells += ['--', '--', f'{v["accuracy"] * 100:.1f}' if v else '--', '--', '--']
            L.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    for ds in dss:
        n = [(r['n_masked'], r['n_test']) for v in (bg.get((ds, m), []) for m in DEFAULT_BENCHMARK) for r in v]
        if n:
            L.append(f'% {ds}: {n[0][0]} of {n[0][1]} test images have a leaf mask (uniform background)')
    with open(os.path.join(out_dir, 'background.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print("background.tex written")


def val_macro_f1(run_dir):
    """Validation macro-F1 of the selected checkpoint (model selection on the validation set only)."""
    for p in glob.glob(os.path.join(run_dir, '*_summary.json')):
        with open(p) as f:
            v = json.load(f).get('best_val', {})
        if isinstance(v.get('macro_f1'), (int, float)):
            return float(v['macro_f1'])
    return None


ENSEMBLE_SEEDS = (0, 1, 2)


def ensemble_candidates(main_items, ds, seed=0):
    """Base runs of one seed of a dataset sorted by validation macro-F1 -> (all, best of each model category), each a list of
    (validation macro-F1, model, run)."""
    cand = []
    for (d_, model, size, tag), runs in main_items:
        r = next((x for x in runs if x['_seed'] == seed), None) if d_ == ds and size == 224 else None
        v = val_macro_f1(r['_dir']) if r else None
        if v is not None and os.path.exists(os.path.join(r['_dir'], 'test_probs.npz')):
            cand.append((v, model, r))
    cand.sort(key=lambda c: -c[0])
    fams, by_family = set(), []
    for c in cand:
        if family(c[1]) not in fams:
            fams.add(family(c[1]))
            by_family.append(c)
    return cand, by_family


def ci_txt(e, digits=1):
    m, lo, hi = (v * 100 for v in e)
    return f'{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'.replace('-', '$-$')


def write_paired_ci(main_items, refs, out_dir, n_boot=2000):
    """paired_ci.tex: difference in macro-F1 and accuracy of every model to the reference of its dataset (best
    mean validation macro-F1), mean over seeds, with a 95 % paired bootstrap interval over test images and seeds."""
    rng = np.random.default_rng(0)
    by = {(k[0], k[1]): v for k, v in main_items if k[2] == 224}
    dss = [ds for ds in ('TeaLeafBD', 'TeaSickness', 'TeaDisease') if ds in refs]
    cells, ref_val = {}, {}
    for ds in dss:
        ref_p = [x for x in (load_probs(r['_dir']) for r in by.get((ds, refs[ds]), [])) if x]
        if not ref_p:
            continue
        y, k = ref_p[0][0], len(ref_p[0][2])
        pb = [x[1].argmax(1) for x in ref_p]
        ref_val[ds] = np.mean([macro_f1_fast(y, p, k) for p in pb])
        for model in DEFAULT_BENCHMARK:
            pa = [x for x in (load_probs(r['_dir']) for r in by.get((ds, model), [])) if x]
            if model == refs[ds] or not pa or any(not np.array_equal(x[0], y) for x in pa):
                continue
            cells[(ds, model)] = boot_diff(y, [x[1].argmax(1) for x in pa], pb, k, rng, n_boot)
            f, c = cells[(ds, model)]
            print(f"  paired {ds:<12} {model:<18} dF1={f[0] * 100:+.2f} [{f[1] * 100:+.2f}, {f[2] * 100:+.2f}]  "
                  f"dAcc={c[0] * 100:+.2f} [{c[1] * 100:+.2f}, {c[2] * 100:+.2f}]")
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         f'% model minus reference (best mean validation macro-F1), percentage points, mean over seeds; 95 % paired '
         f'bootstrap interval over test images and seeds ({n_boot} resamples)',
         '\\begin{tabular}{l' + 'cc' * len(dss) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{DS_LABEL[ds]} (ref.\\ {display_name(refs[ds])})}}'
                            for ds in dss) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(dss))),
         'Model & ' + ' & '.join(['$\\Delta$F1$_m$ [95\\,\\% CI] & $\\Delta$Acc [95\\,\\% CI]'] * len(dss)) + ' \\\\',
         '\\midrule']
    for model in DEFAULT_BENCHMARK:
        row = []
        for ds in dss:
            if model == refs[ds]:
                row.append(f'\\multicolumn{{2}}{{c}}{{reference (F1$_m$ {ref_val.get(ds, float("nan")) * 100:.1f})}}')
            elif (ds, model) in cells:
                row += [ci_txt(e) for e in cells[(ds, model)]]
            else:
                row += ['--', '--']
        L.append(f'{display_name(model)} & ' + ' & '.join(row) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, 'paired_ci.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"paired_ci.tex: {len(cells)} comparisons")


def calib(probs, labels):
    from utils.metrics import expected_calibration_error
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    return expected_calibration_error(probs, labels), nll, brier(probs, labels)


def write_calibration(main_items, out_dir):
    """calibration.tex: ECE (15 equal-width confidence bins), NLL and Brier score of the test predictions, mean over
    seeds, and of the top-3 soft-voting ensemble (members chosen on the validation set within each seed)."""
    by = {(k[0], k[1]): v for k, v in main_items if k[2] == 224}
    dss = [ds for ds in ('TeaLeafBD', 'TeaSickness', 'TeaDisease') if any(k[0] == ds for k in by)]

    def cells(vals):
        if not vals:
            return ['--'] * 3
        v = np.array(vals)
        m, sd = v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros(3))
        pmx = lambda i, s, d: f'{m[i] * s:.{d}f}' + (f'{{\\scriptsize$\\pm${sd[i] * s:.{d}f}}}' if len(v) > 1 else '')  # noqa: E731
        return [pmx(0, 100, 1), pmx(1, 1, 2), pmx(2, 1, 3)]

    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% ECE (%, 15 equal-width confidence bins), NLL and Brier score on the test set; mean +- sd over seeds',
         '\\begin{tabular}{l' + 'ccc' * len(dss) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{DS_LABEL[ds]}}}' for ds in dss) + ' \\\\',
         ''.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(dss))),
         'Model & ' + ' & '.join(['ECE & NLL & Brier'] * len(dss)) + ' \\\\', '\\midrule']
    for model in DEFAULT_BENCHMARK:
        row = []
        for ds in dss:
            row += cells([calib(p[1], p[0]) for p in (load_probs(r['_dir']) for r in by.get((ds, model), [])) if p])
        L.append(f'{display_name(model)} & ' + ' & '.join(row) + ' \\\\')
    row = []
    for ds in dss:
        vals = []
        for seed in ENSEMBLE_SEEDS:
            cand, _ = ensemble_candidates(main_items, ds, seed)
            zs = [load_probs(c[2]['_dir']) for c in cand[:3]]
            if len(zs) == 3:
                vals.append(calib(np.mean([z[1] for z in zs], 0), zs[0][0]))
        row += cells(vals)
    L += ['\\midrule', 'Ensemble (top-3, soft voting) & ' + ' & '.join(row) + ' \\\\', '\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, 'calibration.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print("calibration.tex written")


def write_ensembles(main_items, out_dir):
    """ensembles.tex: soft voting (mean of the test probabilities) of the base models of one seed, members chosen by
    their validation macro-F1 within that seed: the best three models, and the best model of three different
    model categories; next to the best single model by validation. Mean +- sd over seeds; in brackets the number of seeds in
    which a model was a member."""
    from utils.metrics import classification_metrics
    L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
         '% members chosen on the validation set within each seed; Acc / F1_m in %, mean +- sd over seeds; in brackets: '
         'number of seeds in which a model was a member',
         '\\begin{tabular}{lp{0.55\\linewidth}cc}', '\\toprule', 'Dataset & Model / members & Acc & F1$_m$ \\\\', '\\midrule']
    done = 0
    for ds in ('TeaLeafBD', 'TeaSickness', 'TeaDisease'):
        rows = defaultdict(list)
        for seed in ENSEMBLE_SEEDS:
            cand, by_family = ensemble_candidates(main_items, ds, seed)
            if len(cand) < 3:
                continue
            for label, members in (('best single (validation)', cand[:1]), ('top-3', cand[:3]),
                                   ('top of 3 families', by_family[:3])):
                zs = [np.load(os.path.join(m[2]['_dir'], 'test_probs.npz')) for m in members]
                if any(not np.array_equal(z['labels'], zs[0]['labels']) for z in zs):
                    raise SystemExit(f"{ds}: test images differ between ensemble members")
                m = classification_metrics(np.mean([z['probs'] for z in zs], axis=0), zs[0]['labels'],
                                           [str(c) for c in zs[0]['classes']])
                names = ', '.join(display_name(x[1]) for x in members)
                rows[label].append((m['accuracy'], m['macro_f1'], [x[1] for x in members]))
                print(f"  ensemble {ds:<12} s{seed} {label:<26} acc={m['accuracy'] * 100:.2f} "
                      f"macro-F1={m['macro_f1'] * 100:.2f}  [{names}]")
        if not rows:
            continue
        if done:
            L.append('\\midrule')
        for i, (label, res) in enumerate(rows.items()):
            count = defaultdict(int)
            for r in res:
                for x in r[2]:
                    count[x] += 1
            names = ', '.join(f'{display_name(x)} ({n})' for x, n in sorted(count.items(), key=lambda c: (-c[1], c[0])))
            a, f = (np.array([r[j] for r in res]) for j in (0, 1))
            sd = (lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            L.append(f'{DS_LABEL[ds] if i == 0 else ""} & {label}: {names} & '
                     f'{fmt((a.mean(), sd(a), len(a)), digits=1)} & {fmt((f.mean(), sd(f), len(f)), digits=1)} \\\\')
        done += 1
    L += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, 'ensembles.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"ensembles.tex: {done} datasets")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    p.add_argument('--out_dir', default=os.path.join(ROOT, 'paper', 'tables'))
    p.add_argument('--suffix', default='')
    p.add_argument('--tag', default=None, help='only runs with this tag')
    p.add_argument('--img_size', type=int, default=None, help='only runs with this input size')
    p.add_argument('--ref', default=None, help='reference model of the McNemar tests (default: best macro-F1)')
    p.add_argument('--mcnemar_datasets', nargs='+', default=['TeaLeafBD', 'TeaSickness'],
                   help='datasets with McNemar tests (TeaDisease: seed spread instead, see the main table)')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    groups = load_runs(a.results_dir, a.suffix)
    groups = {k: v for k, v in groups.items() if (a.tag is None or k[3] == a.tag) and
              (a.img_size is None or k[2] == a.img_size)}
    if not groups:
        print(f"no finished runs with test_results{a.suffix}.json in {a.results_dir}")
        return
    items = sorted(groups.items(), key=sort_key)

    # ---------------------------------------------------------------- summary.csv
    keys = [m for m, _ in METRICS] + ['weighted_f1', 'top3_accuracy', 'macro_auc', 'nll', 'ece',
                                      'leaf_energy_share', 'leaf_focus_ratio',
                                      'pp_leaf_energy_share', 'pp_leaf_focus_ratio',
                                      'occ_leaf_energy_share', 'occ_leaf_focus_ratio']
    with open(os.path.join(a.out_dir, f'summary{a.suffix}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'model', 'img_size', 'tag', 'n_seeds'] + [f'{k}_{s}' for k in keys for s in ('mean', 'std')]
                   + ['params_m', 'gflops', 'fps'])
        for (ds, model, size, tag), runs in items:
            row = [ds, model, size, tag, len(runs)]
            for k in keys:
                s = stat(runs, k)
                row += [f'{s[0]:.5f}', f'{s[1]:.5f}'] if s else ['', '']
            for k in ('cx_params_m', 'cx_gflops', 'cx_fps'):
                s = stat(runs, k)
                row.append(f'{s[0]:.3f}' if s else '')
            w.writerow(row)

    # ---------------------------------------------------------------- main table per dataset
    # main / McNemar / efficiency tables: one tag only (default base); the leakage runs have their own table
    main_tag = a.tag or 'base'
    main_items = [(k, v) for k, v in items if k[3] == main_tag]
    datasets = sorted({k[0] for k, _ in main_items})
    mc = {}                                   # dataset -> (reference model, McNemar tests) for the combined table
    refs = {}                                 # dataset -> reference model (best mean validation macro-F1)
    for ds in datasets:
        rows = [(k, v) for k, v in main_items if k[0] == ds]
        best = {m: max((stat(v, m) or (-1, 0, 0))[0] for _, v in rows) for m, _ in METRICS}
        # macro recall equals balanced accuracy and every run has size 224: neither gets a column here (summary.csv has them)
        shown = [x for x in METRICS if x[0] != 'macro_recall']
        L = ['% generated by tools/make_paper_tables.py -- do not edit by hand',
             '\\begin{tabular}{l' + 'c' * (len(shown) + 1) + '}', '\\toprule',
             'Model & ' + ' & '.join(h for _, h in shown) + ' & Params (M) \\\\', '\\midrule']
        last_fam = None
        for (_, model, size, tag), runs in rows:
            fam = family(model)
            if last_fam is not None and fam != last_fam:
                L.append('\\midrule')
            last_fam = fam
            cells = []
            for m, _ in shown:
                s = stat(runs, m)
                cells.append(fmt(s, bold=s is not None and abs(s[0] - best[m]) < 1e-12))
            pm = stat(runs, 'cx_params_m')
            name = display_name(model) + (f' ({tag})' if tag != 'base' else '')
            L.append(f'{name} & ' + ' & '.join(cells) + f" & {pm[0]:.1f} \\\\" if pm else
                     f'{name} & ' + ' & '.join(cells) + ' & -- \\\\')
        L += ['\\bottomrule', '\\end{tabular}']
        with open(os.path.join(a.out_dir, f'main_{ds}{a.suffix}.tex'), 'w') as f:
            f.write('\n'.join(L) + '\n')

        # Reference of the paired comparisons: the model with the best mean VALIDATION macro-F1 (fixed without
        # looking at the test set). McNemar per seed (same seed of both models), Holm-adjusted per seed.
        ref = normalize_name(a.ref) if a.ref else max(rows, key=lambda kv: mean_val_f1(kv[1]))[0][1]
        refs[ds] = ref
        if ds not in a.mcnemar_datasets:
            continue
        ref_runs = [v for k, v in rows if k[1] == ref]
        if not ref_runs:
            continue
        ref_by_seed = {r['_seed']: r for r in ref_runs[0]}
        per_seed = defaultdict(dict)               # seed -> model -> (n01, n10, p)
        for (_, model, size, tag), runs in rows:
            if model == ref:
                continue
            for other in runs:
                if other['_seed'] not in ref_by_seed:
                    continue
                ca = read_correct(ref_by_seed[other['_seed']]['_dir'], a.suffix)
                cb = read_correct(other['_dir'], a.suffix)
                if ca and cb:
                    shared = sorted(set(ca) & set(cb))
                    per_seed[other['_seed']][model] = mcnemar([ca[x] for x in shared], [cb[x] for x in shared])
        adj = {}                                   # (seed, model) -> Holm-adjusted p
        for seed, res in per_seed.items():
            models = sorted(res)
            adj.update({(seed, m): ph for m, ph in zip(models, holm([res[m][2] for m in models]))})
        M = ['% McNemar (exact) vs ' + display_name(ref) + ' (best mean validation macro-F1), seed 0; n01: ref right / '
             'other wrong, n10: the reverse; p_Holm: Holm-adjusted over the comparisons of this seed; Sig.: seeds with '
             'p_Holm < 0.05', '\\begin{tabular}{lrrrrc}', '\\toprule',
             'Model & $n_{01}$ & $n_{10}$ & $p$ & $p_\\mathrm{Holm}$ & Sig. \\\\', '\\midrule']
        tests = []
        for (_, model, size, tag), runs in rows:
            seeds = sorted(s for s in per_seed if model in per_seed[s])
            if model == ref or not seeds:
                continue
            s0 = 0 if 0 in seeds else seeds[0]
            n01, n10, pv = per_seed[s0][model]
            tests.append((model, n01, n10, pv, adj[(s0, model)],
                          sum(adj[(s, model)] < 0.05 for s in seeds), len(seeds)))
        mc[ds] = (ref, tests)
        for model, n01, n10, pv, ph, nsig, ns in tests:
            M.append(f'{display_name(model)} & {n01} & {n10} & {ptxt(pv)} & {ptxt(ph)} & {nsig}/{ns} \\\\')
        M += ['\\bottomrule', '\\end{tabular}']
        with open(os.path.join(a.out_dir, f'mcnemar_{ds}{a.suffix}.tex'), 'w') as f:
            f.write('\n'.join(M) + '\n')
    if mc:
        write_mcnemar_combined(mc, a.out_dir, a.suffix)
    if not a.suffix:
        write_paired_ci(main_items, refs, a.out_dir)
        write_calibration(main_items, a.out_dir)

    # ---------------------------------------------------------------- efficiency (one row per model / size)
    eff = {}       # cost of the TeaLeafBD model where there is one (7-class head; the head size changes params slightly)
    for (ds, model, size, tag), runs in sorted(main_items, key=lambda kv: kv[0][0] != 'TeaLeafBD'):
        eff.setdefault((model, size), runs)
    E = ['\\begin{tabular}{lcrrrr}', '\\toprule',
         'Model & Size & Params (M) & GFLOPs & FPS & Latency (ms) \\\\', '\\midrule']
    for (model, size), runs in sorted(eff.items(), key=lambda kv: (stat(kv[1], 'cx_params_m') or (0,))[0]):
        vals = [stat(runs, k) for k in ('cx_params_m', 'cx_gflops', 'cx_fps', 'cx_latency_ms')]
        E.append(f'{display_name(model)} & {size} & ' + ' & '.join(f'{v[0]:.1f}' if v else '--' for v in vals) + ' \\\\')
    E += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(a.out_dir, 'efficiency.tex'), 'w') as f:
        f.write('\n'.join(E) + '\n')

    if not a.suffix:
        write_leakage(groups, a.out_dir)
        write_leakage_decomp(groups, a.out_dir)
        write_leakage_perclass(groups, a.out_dir)
        write_cross(a.results_dir, a.out_dir)
        write_leaf_focus(main_items, a.out_dir)
        write_background(main_items, a.results_dir, a.out_dir)
        write_ensembles(main_items, a.out_dir)

    print(f"{len(groups)} configurations from {sum(len(v) for v in groups.values())} runs -> {a.out_dir}")
    for (ds, model, size, tag), runs in items:
        s = stat(runs, 'macro_f1')
        acc = stat(runs, 'accuracy')
        print(f"  {ds:<20} {model:<20} {size:>4} {tag:<10} seeds={len(runs)} "
              f"acc={acc[0] * 100:.2f} macro-F1={s[0] * 100:.2f}±{s[1] * 100:.2f}")


if __name__ == '__main__':
    main()
