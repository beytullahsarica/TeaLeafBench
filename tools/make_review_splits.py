"""
Alternative splits for the sensitivity analyses of the paper. Uses only the manifests of the prepared datasets; the
images stay where they are and the main splits are not changed. Writes splits_review/<name>.json (--split_file of
train.py) and splits_review/summary.json.

  TeaDisease_grouped   stratified 70/15/15 with same-leaf (duplicate) groups as units, sessions ignored: separates the
                       duplicates found by the detector from the shift between sessions
  TeaDisease_p1..p3    other assignments of whole sessions to train / val / test, near the target proportions
  TeaDisease_gap30     new session after a pause of >= 30 frame numbers instead of 12
  TeaDisease_f25, f50  main session split with 25 / 50 % of the training units per class (learning curve)
  TeaLeafBD_eq         training set reduced to the size of the TeaDisease training set; shared classes capped at their
                       TeaDisease counts
  TeaLeafBD_ratio      same, and the three shared classes also in the proportions of the TeaDisease training set
                       (not used in the paper)
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'splits_review')
FRAME = re.compile(r'IMG_(\d+)')
RATIOS = (0.70, 0.15, 0.15)
SPLITS = ('train', 'val', 'test')
CUTS = {'healthy': (1330,)}              # the session divided at a change of leaf (checked by eye for the main split)
SHARED = {'healthy': 'healthy', 'red_spider': 'red_spider', 'tea_mosquito_bug': 'helopeltis'}   # TeaDisease -> TeaLeafBD


def read_manifest(dataset):
    with open(os.path.join(ROOT, 'datasets_prepared', dataset, 'manifest.csv'), newline='') as f:
        return [r for r in csv.DictReader(f) if r['split'] in SPLITS]


def frame_no(rel):
    return int(FRAME.search(os.path.basename(rel)).group(1))


def sessions_of_class(rows, gap, cuts=()):
    """Rows of one class -> list of sessions (lists of rows): a new session after a pause of >= gap frame numbers or at
    a cut; sessions that share a same-leaf group are merged, so that no group crosses a session."""
    rows = sorted(rows, key=lambda r: frame_no(r['file']))
    sid, cur, end = {}, -1, None
    for r in rows:
        n = frame_no(r['file'])
        if end is None or n - end >= gap or any(end < c <= n for c in cuts):
            cur += 1
        sid[r['file']], end = cur, n
    parent = list(range(cur + 1))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    first = {}
    for r in rows:
        g = r.get('dup_group', '')
        if g in ('', '-1'):
            continue
        if g in first:
            parent[find(sid[r['file']])] = find(first[g])
        else:
            first[g] = sid[r['file']]
    out = defaultdict(list)
    for r in rows:
        out[find(sid[r['file']])].append(r)
    return [out[k] for k in sorted(out)]


def units_of_class(rows):
    """Same-leaf groups and single images."""
    out = defaultdict(list)
    for r in rows:
        g = r.get('dup_group', '')
        out[r['file'] if g in ('', '-1') else f'g{g}'].append(r)
    return [out[k] for k in sorted(out)]


def to_split(classes, parts):
    return {'classes': classes, **{s: [[r['file'], r['class']] for r in sorted(parts[s], key=lambda r: r['file'])]
                                   for s in SPLITS}, 'ratios': list(RATIOS), 'dedup': 'review'}


def assign_sessions(sessions, rng, avoid_test=(), n_try=30000, top=100):
    """One assignment of whole sessions to the three sets, near the target proportions, chosen among the best: val and
    test not empty, training keeps at least half of the images, test set different from those in avoid_test."""
    sizes = np.array([len(s) for s in sessions])
    target = np.array(RATIOS) * sizes.sum()
    best = {}
    for _ in range(n_try):
        a = rng.integers(0, 3, len(sessions))
        n = np.array([sizes[a == k].sum() for k in range(3)])
        if n[1] == 0 or n[2] == 0 or n[0] < 0.5 * sizes.sum():
            continue
        key = tuple(a.tolist())
        test = frozenset(np.nonzero(a == 2)[0].tolist())
        if test in avoid_test:
            continue
        best[key] = float(np.abs(n - target).sum())
    if not best:                    # too few sessions for another test set: repeat an earlier one
        return assign_sessions(sessions, rng, (), n_try, top) if avoid_test else None
    keys = sorted(best, key=best.get)[:top]
    w = np.exp(-np.array([best[k] for k in keys]) / max(sizes.sum() * 0.05, 1.0))
    return np.array(keys[int(rng.choice(len(keys), p=w / w.sum()))])


def session_partition(rows, classes, gap, seed, avoid=None):
    """avoid: {class: set of test-session index sets} is updated, so that repeated partitions differ where possible"""
    rng = np.random.default_rng(seed)
    parts = {s: [] for s in SPLITS}
    avoid = {} if avoid is None else avoid
    info = {}
    for c in classes:
        sess = sessions_of_class([r for r in rows if r['class'] == c], gap, CUTS.get(c, ()))
        a = assign_sessions(sess, rng, avoid.setdefault(c, set()))
        if a is None:
            raise SystemExit(f'{c}: {len(sess)} sessions cannot be assigned')
        avoid[c].add(frozenset(np.nonzero(a == 2)[0].tolist()))
        for k, s in zip(a, sess):
            parts[SPLITS[k]].extend(s)
        info[c] = {'sessions': len(sess), **{SPLITS[k]: int((a == k).sum()) for k in range(3)}}
    return parts, info


def released_test_sets(rows, classes, gap):
    """Test-session index sets of the main session split, so that the new partitions differ from it."""
    avoid = {}
    for c in classes:
        sess = sessions_of_class([r for r in rows if r['class'] == c], gap, CUTS.get(c, ()))
        avoid[c] = {frozenset(i for i, s in enumerate(sess) if s[0]['split'] == 'test')}
    return avoid


def grouped_split(rows, classes, seed=42):
    rng = np.random.default_rng(seed)
    parts = {s: [] for s in SPLITS}
    for c in classes:
        units = units_of_class([r for r in rows if r['class'] == c])
        n = sum(len(u) for u in units)
        got = np.zeros(3)
        for i in rng.permutation(len(units)):
            k = int(np.argmax(np.array(RATIOS) * n - got))      # the set furthest below its target
            parts[SPLITS[k]].extend(units[i])
            got[k] += len(units[i])
    return parts


def fraction_split(rows, classes, frac, seed=0):
    rng = np.random.default_rng(seed)
    parts = {s: [r for r in rows if r['split'] == s] for s in SPLITS}
    train = []
    for c in classes:
        units = units_of_class([r for r in parts['train'] if r['class'] == c])
        want, n = max(8, frac * sum(len(u) for u in units)), 0
        for i in rng.permutation(len(units)):
            if n >= want:
                break
            train.extend(units[i])
            n += len(units[i])
    parts['train'] = train
    return parts


def subset_train(rows, classes, counts, seed=0):
    rng = np.random.default_rng(seed)
    parts = {s: [r for r in rows if r['split'] == s] for s in SPLITS}
    train = []
    for c in classes:
        x = [r for r in parts['train'] if r['class'] == c]
        k = min(int(round(counts[c])), len(x))
        train.extend(x[i] for i in rng.permutation(len(x))[:k])
    parts['train'] = train
    return parts


def main():
    os.makedirs(OUT, exist_ok=True)
    td = read_manifest('TeaDisease')
    td_classes = sorted({r['class'] for r in td})
    summary = {}

    def save(name, classes, parts, extra=None):
        with open(os.path.join(OUT, f'{name}.json'), 'w') as f:
            json.dump(to_split(classes, parts), f)
        summary[name] = {s: dict(sorted(Counter(r['class'] for r in parts[s]).items())) for s in SPLITS}
        if extra:
            summary[name]['sessions'] = extra
        print(name, {s: len(parts[s]) for s in SPLITS}, flush=True)

    save('TeaDisease_grouped', td_classes, grouped_split(td, td_classes))
    avoid = released_test_sets(td, td_classes, 12)
    for p in (1, 2, 3):
        parts, info = session_partition(td, td_classes, 12, p, avoid)
        save(f'TeaDisease_p{p}', td_classes, parts, info)
    parts, info = session_partition(td, td_classes, 30, 30)
    save('TeaDisease_gap30', td_classes, parts, info)
    for frac, tag in ((0.25, 'f25'), (0.50, 'f50')):
        save(f'TeaDisease_{tag}', td_classes, fraction_split(td, td_classes, frac))

    # paper dataset reduced to the size and the shared-class counts of the field training set
    bd = read_manifest('TeaLeafBD')
    bd_classes = sorted({r['class'] for r in bd})
    n_td = Counter(r['class'] for r in td if r['split'] == 'train')
    n_bd = Counter(r['class'] for r in bd if r['split'] == 'train')
    total = sum(n_td.values())
    shared_bd = {v: k for k, v in SHARED.items()}
    capped = {c: min(n_bd[c], n_td[shared_bd[c]]) for c in shared_bd}
    others = [c for c in bd_classes if c not in shared_bd]
    left = total - sum(capped.values())
    scale = left / sum(n_bd[c] for c in others)
    save('TeaLeafBD_eq', bd_classes, subset_train(bd, bd_classes, {**capped, **{c: n_bd[c] * scale for c in others}}))
    # three shared classes in the field proportions, limited by the largest class that can be matched
    k = min(n_bd[c] / n_td[shared_bd[c]] for c in shared_bd)
    ratio = {c: n_td[shared_bd[c]] * k for c in shared_bd}
    other_share = sum(n_td[c] for c in td_classes if c not in SHARED) / total
    n_other = sum(ratio.values()) * other_share / (1 - other_share)
    save('TeaLeafBD_ratio', bd_classes, subset_train(bd, bd_classes, {**ratio, **{c: n_other / len(others) for c in others}}))

    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
