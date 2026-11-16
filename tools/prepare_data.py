"""
Prepare the three tea leaf datasets for classification: clean, de-duplicated, harmonised class names, and a
fixed stratified train / val / test split with the same class proportions in every split.

Standalone: needs only Python >= 3.8, numpy and Pillow. Runs on a laptop (multiprocessing, a few minutes).

  python tools/prepare_data.py --src datasets --out datasets_prepared            # all three, 70/15/15, seed 42
  python tools/prepare_data.py --src datasets --out datasets_prepared --dry_run  # tables + checks only, no copying
  # effect-of-leakage experiment (same cleaned images, leaky assignments; never used for the main results):
  python tools/prepare_data.py --src datasets --out datasets_leaky_random --split_mode random
  python tools/prepare_data.py --src datasets --out datasets_leaky_official --split_mode official

Input (--src), as downloaded:
  teaLeafBD/<n. class>/*.jpg                         class folders, no split
  tea sickness dataset/<class>/*.jpg                 class folders, no split
  tea_disease_dataset/{train,valid,test}/{images,labels}   YOLO detection export (Roboflow), official split

Output (--out), ready for train.py (pre-split layout, files copied unchanged unless --max_side is given):
  <out>/TeaLeafBD/{train,val,test}/<class>/*.jpg
  <out>/TeaSickness/{train,val,test}/<class>/*.jpg
  <out>/TeaDisease/{train,val,test}/<class>/*.jpg
  <out>/<Dataset>/manifest.csv       one row per image: split, class, source file, md5, duplicate group
  <out>/<Dataset>/removed.csv        every image that was left out and why
  <out>/_checks/*.jpg                image pairs judged duplicates / closest non-duplicates (check by eye!)
  <out>/prepare_report.md            all tables of this run (counts per class and split, duplicates, removals)

What it does
  1. Class names are harmonised across datasets (e.g. "gray light" -> gray_blight, "Tea leaf" -> healthy), so
     cross-dataset tests are possible later.
  2. TeaDisease (detection -> classification): Roboflow stored every photo 3-4 times as byte/pixel-identical
     copies (IMG_0151_JPG_jpg.rf.<hash>.jpg); one copy per photo is kept. The image label is the class of its
     boxes; images whose boxes carry different classes are left out. Classes with too few photos are dropped.
     Its official split is NOT kept by default: near-identical shots of the same scene sit in train and test.
  3. Duplicates: exact (MD5) and near duplicates (re-saved copies, burst shots of the same leaf). Images are
     compared on the LEAF CROP (leaf cut out of the uniform paper background): whole-image hashes / thumbnails
     match different leaves because the background dominates. Duplicate groups are placed in ONE split (no
     leakage); duplicates whose copies carry different labels are removed (label noise).
  4. Split: stratified by class, 70/15/15 (default), seed 42; duplicate groups are kept together. The split is
     the same on every machine (sorted relative paths + seed).
  5. TeaDisease (field photos with frame numbers, IMG_<frame>): consecutive frames are often the same leaf seen from
     a slightly different position, which the leaf-crop test cannot detect (the background changes). Each class is
     split by shooting session instead (session_split): whole sessions per split, like patients in medical imaging.
     Check _checks/TeaDisease_split_cuts.jpg (frames on both sides of every session boundary) by eye.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from PIL import Image

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# folder name in --src -> (output name, {original class name: harmonised class name})
FOLDER_DATASETS = {
    'teaLeafBD': ('TeaLeafBD', {
        '1. Tea algal leaf spot': 'algal_leaf_spot', '2. Brown Blight': 'brown_blight',
        '3. Gray Blight': 'gray_blight', '4. Helopeltis': 'helopeltis', '5. Red spider': 'red_spider',
        '6. Green mirid bug': 'green_mirid_bug', '7. Healthy leaf': 'healthy'}),
    'tea sickness dataset': ('TeaSickness', {
        'algal leaf': 'algal_leaf_spot', 'Anthracnose': 'anthracnose', 'bird eye spot': 'bird_eye_spot',
        'brown blight': 'brown_blight', 'gray light': 'gray_blight',   # "gray light" = gray blight (dataset readme)
        'healthy': 'healthy', 'red leaf spot': 'red_leaf_spot', 'white spot': 'white_spot'}),
}
YOLO_DATASET = ('tea_disease_dataset', 'TeaDisease', [
    'black_rot', 'brown_blight', 'leaf_rust', 'red_spider', 'tea_mosquito_bug', 'healthy', 'white_spot',
    'disease'])            # data.yaml order: Black rot, Brown blight, Leaf rust, Red Spider infested, Tea Mosquito
                           # bug infested, Tea leaf, White spot, disease
YOLO_SPLITS = {'train': 'train', 'valid': 'val', 'test': 'test'}
# field photos with frame numbers (IMG_<frame>): split by shooting session, see session_split()
CAPTURE_ORDER_DATASETS = {'TeaDisease'}
FRAME = re.compile(r'IMG_(\d+)')
# sessions cut by hand at a change of leaf (checked by eye: frame 1329 | 1330 show different leaves); the parts may
# go to val and test, never to train. Needed because healthy has only two large sessions.
SESSION_CUTS = {'TeaDisease': {'healthy': [1329]}}


# ----------------------------------------------------------------------------- image features
def leaf_box(a, thr=45.0):
    """Bounding box of the leaf: pixels that differ from the background colour (median of the image border).
    The datasets show one leaf on uniform paper; whole-image comparisons are dominated by that background and
    match different leaves, so duplicates are compared on the leaf crop."""
    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                             a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    mask = np.linalg.norm(a - np.median(border, 0), axis=2) > thr
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:                                  # no clear foreground (e.g. field photo): whole image
        return 0, a.shape[0], 0, a.shape[1]
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)   # percentiles: ignore specks of dirt on the paper
    x0, x1 = np.percentile(xs, [1, 99]).astype(int)
    return y0, y1 + 1, x0, x1 + 1


def features(path):
    """md5, difference hash + aspect ratio + mean-free 48x48 RGB thumbnail of the leaf crop, image size.
    A string (the error) for unreadable files."""
    try:
        with open(path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        im = Image.open(path)
        size = im.size
        im.draft('RGB', (256, 256))                   # fast JPEG decoding at reduced size
        im = im.convert('RGB')
        im.thumbnail((256, 256))
        y0, y1, x0, x1 = leaf_box(np.asarray(im, dtype=np.float32))
        crop = im.crop((x0, y0, x1, y1))
        g = np.asarray(crop.convert('L').resize((9, 8), Image.BILINEAR), dtype=np.int16)
        dh = int(np.packbits((g[:, 1:] > g[:, :-1]).ravel()).view('>u8')[0])
        t = np.asarray(crop.resize((48, 48), Image.BILINEAR), dtype=np.float32)
        t -= t.mean(axis=(0, 1), keepdims=True)       # robust to brightness / white-balance changes
        return md5, dh, t.ravel().astype(np.float16), size, (x1 - x0) / max(y1 - y0, 1)
    except Exception as e:  # noqa: BLE001
        return str(e)


def compute_features(paths, workers):
    with Pool(workers) as pool:
        return pool.map(features, paths, chunksize=32)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def find_duplicates(feats, labels, hash_dist, dup_thr, group_thr):
    """
    feats: list of (md5, dhash, crop thumb, size, aspect). Pairs are candidates if their crop hashes are close
    and the leaf aspect ratios agree; then the mean absolute difference of the crop thumbnails (0-255) decides:
      <= dup_thr                    duplicate (same photo / burst shot of the same leaf), any labels
      <= group_thr and same label   probably the same leaf: kept in the same split (never a label conflict)
    Returns two union-finds (duplicates, split groups), the duplicate pairs, the same-split-only pairs.
    """
    n = len(feats)
    dup, grp = UnionFind(n), UnionFind(n)
    first = {}
    for i, f in enumerate(feats):
        if f[0] in first:
            dup.union(first[f[0]], i)
            grp.union(first[f[0]], i)
        else:
            first[f[0]] = i
    H = np.array([f[1] for f in feats], dtype=np.uint64)
    T = np.stack([f[2] for f in feats]).astype(np.float32)
    AR = np.log(np.array([f[4] for f in feats], dtype=np.float64))
    dups, groups = [], []
    for i in range(n - 1):
        x = np.bitwise_xor(H[i + 1:], H[i])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        cand = np.nonzero((d <= hash_dist) & (np.abs(AR[i + 1:] - AR[i]) < 0.15))[0]
        if not len(cand):
            continue
        js = cand + i + 1
        mae = np.abs(T[js] - T[i]).mean(1)
        for j, m, hd in zip(js, mae, d[cand]):
            j = int(j)
            if feats[i][0] == feats[j][0]:
                continue                                    # exact duplicate, already joined
            if m <= dup_thr:
                dup.union(i, j)
                grp.union(i, j)
                dups.append((float(m), int(hd), i, j))
            elif m <= group_thr and labels[i] == labels[j]:
                grp.union(i, j)
                groups.append((float(m), int(hd), i, j))
    return dup, grp, sorted(dups), sorted(groups)


# ----------------------------------------------------------------------------- split
def stratified_split(units, ratios, seed):
    """
    units: list of (unit id, class, n_images). Every unit (duplicate group) goes to one split; per class the
    units are shuffled with the seed and each is given to the split that is furthest below its target share.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for u, c, n in sorted(units):
        by_class[c].append((u, n))
    assign = {}
    for c in sorted(by_class):
        us = by_class[c][:]
        rng.shuffle(us)
        us.sort(key=lambda x: -x[1])        # big groups first (stable: ties keep the shuffled order)
        total = sum(n for _, n in us)
        target = [r * total for r in ratios]
        count = [0] * len(ratios)
        for u, n in us:
            k = max(range(len(ratios)), key=lambda s: (target[s] - count[s]) / max(target[s], 1e-9))
            assign[u] = k
            count[k] += n
    return assign


def frame_no(rel):
    m = FRAME.search(os.path.basename(rel))
    return int(m.group(1)) if m else None


def capture_sessions(units, gap):
    """units of one class [(first frame, last frame, n, unit)] -> shooting sessions (lists of units, capture order);
    a new session starts after a pause of >= gap frame numbers"""
    sessions, end = [], None
    for x in sorted(units):
        if end is None or x[0] - end >= gap:
            sessions.append([])
        sessions[-1].append(x)
        end = x[1] if end is None or x[0] - end >= gap else max(end, x[1])
    return sessions


def session_split(units, ratios, gap, cuts=None, min_heldout=0.25):
    """
    units: list of (unit id, class, first frame, last frame, n_images). Field photos with frame numbers: a leaf is
    photographed several times in a row and may reappear later in the same session, so no
    cut inside a session is safe. The images are grouped into shooting sessions (a new session starts after a pause
    of >= gap frame numbers) and whole sessions are assigned to the splits, as patients are in medical imaging. Per
    class the assignment minimises the largest relative deviation of the train / val / test counts from their
    targets (exhaustive for <= 10 sessions, else greedy + local search); the larger held-out set becomes test.
    cuts {class: [frame, ...]}: a session is cut after these frames (a change of leaf, checked by eye); its parts may
    go to val and test, never to train (a leaf shared by val and test does not inflate the test score).
    A class whose val or test set would get less than min_heldout of its target size is excluded (split None).
    Returns {unit: split index or None}, [(class, text, unit left, unit right)]: one entry per class plus one per
    boundary between consecutive sessions of different splits (for the visual check).
    """
    by_class = defaultdict(list)
    for u, c, f0, f1, n in units:
        by_class[c].append((f0, f1, n, u))
    assign, out = {}, []
    for c in sorted(by_class):
        sessions, heldout = [], []
        for s in capture_sessions(by_class[c], gap):
            parts, cut_here = [s], sorted(f for f in (cuts or {}).get(c, []) if s[0][0] <= f < s[-1][1])
            for f in cut_here:
                last = parts.pop()
                parts += [[x for x in last if x[0] <= f], [x for x in last if x[0] > f]]
            parts = [p for p in parts if p]
            sessions += parts
            heldout += [len(parts) > 1] * len(parts)
        size = np.array([sum(x[2] for x in s) for s in sessions])
        total, k = int(size.sum()), len(sessions)
        tg = np.array([r * total for r in ratios])

        def cost(lab):
            cnt = np.bincount(lab, weights=size, minlength=3)
            if (cnt == 0).any() or any(h and l == 0 for h, l in zip(heldout, lab)):
                return (np.inf, np.inf)
            dev = np.abs(cnt - tg) / tg
            return (dev.max(), dev.sum())

        if 3 ** k <= 60000:
            best = min((np.array(lab) for lab in itertools.product(range(3), repeat=k)), key=cost)
        else:                                    # greedy (largest first, most missing split), then single moves
            best, cnt = np.zeros(k, int), np.zeros(3)
            for i in np.argsort(-size, kind='stable'):
                ok = [1, 2] if heldout[i] else [0, 1, 2]
                best[i] = max(ok, key=lambda s: (tg[s] - cnt[s]) / tg[s])
                cnt[best[i]] += size[i]
            improved = True
            while improved:
                improved = False
                for i in range(k):
                    for s in range(3):
                        lab = best.copy()
                        lab[i] = s
                        if cost(lab) < cost(best):
                            best, improved = lab, True
        cnt = np.bincount(best, weights=size, minlength=3).astype(int)
        if cnt[1] > cnt[2]:                      # the larger held-out set is the test set
            best = np.where(best == 1, 2, np.where(best == 2, 1, best))
            cnt = cnt[[0, 2, 1]]
        spans = ' '.join(f"{s[0][0]}-{max(x[1] for x in s)}:{('train', 'val', 'test')[l]}"
                         for s, l in zip(sessions, best))
        if min(cnt[1] / tg[1], cnt[2] / tg[2]) < min_heldout or not np.isfinite(cost(best)[0]):
            for s in sessions:
                for x in s:
                    assign[x[3]] = None
            out.append((c, f'EXCLUDED, cannot be split by session ({k} sessions, {total} images: {spans})', None, None))
            continue
        for s, l in zip(sessions, best):
            for x in s:
                assign[x[3]] = int(l)
        out.append((c, f'{k} sessions -> {cnt[0]} / {cnt[1]} / {cnt[2]}: {spans}', None, None))
        for (s, l), (t, m) in zip(zip(sessions, best), zip(sessions[1:], best[1:])):
            if l != m:
                out.append((c, f"{('train', 'val', 'test')[l]} | {('train', 'val', 'test')[m]}",
                            max(s, key=lambda x: x[1])[3], min(t, key=lambda x: x[0])[3]))
    return assign, out


# ----------------------------------------------------------------------------- datasets -> records
def is_image(f):
    return f.lower().endswith(IMG_EXT) and not f.startswith('.')


def collect_folder_dataset(src, folder, class_map):
    """records: dict(src=absolute path, rel=path inside the dataset, cls, split=None)"""
    root = os.path.join(src, folder)
    recs, unknown = [], []
    for d in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, d)):
            continue
        if d not in class_map:
            unknown.append(d)
            continue
        for dp, _, fs in os.walk(os.path.join(root, d)):
            for f in sorted(fs):
                if is_image(f):
                    p = os.path.join(dp, f)
                    recs.append({'src': p, 'rel': os.path.relpath(p, root), 'cls': class_map[d], 'split': None})
    if unknown:
        sys.exit(f"[ERROR] {folder}: class folders not in the class map: {unknown}. Add them to FOLDER_DATASETS.")
    return recs


def collect_yolo_dataset(src, folder, names, removed):
    """One record per unique photo (first Roboflow copy); label = the single class of its boxes."""
    root = os.path.join(src, folder)
    recs = []
    for sdir, split in YOLO_SPLITS.items():
        img_dir, lab_dir = os.path.join(root, sdir, 'images'), os.path.join(root, sdir, 'labels')
        groups = defaultdict(list)
        for f in sorted(os.listdir(img_dir)):
            if is_image(f):
                groups[re.sub(r'\.rf\.[0-9a-f]+$', '', os.path.splitext(f)[0])].append(f)
        for base, files in sorted(groups.items()):
            labels = []
            for f in files:
                lp = os.path.join(lab_dir, os.path.splitext(f)[0] + '.txt')
                cls = set()
                if os.path.exists(lp):
                    with open(lp) as fh:
                        cls = {int(line.split()[0]) for line in fh if line.strip()}
                labels.append(frozenset(cls))
            keep, rel = files[0], os.path.join(sdir, 'images', files[0])
            for f in files[1:]:
                removed.append((os.path.join(sdir, 'images', f), split, '', f'roboflow copy of {keep}'))
            if len(set(labels)) > 1:
                removed.append((rel, split, '', 'copies of one photo have different labels'))
                continue
            cls = labels[0]
            if not cls:
                removed.append((rel, split, '', 'no boxes'))
            elif len(cls) > 1:
                removed.append((rel, split, '+'.join(names[c] for c in sorted(cls)), 'boxes of several classes'))
            else:
                recs.append({'src': os.path.join(root, rel), 'rel': rel, 'cls': names[next(iter(cls))],
                             'split': split, 'n_copies': len(files)})
    return recs


# ----------------------------------------------------------------------------- report helpers
def table(counts, classes, splits=('train', 'val', 'test')):
    lines = ['| class | ' + ' | '.join(splits) + ' | total | ' + ' | '.join(f'{s} %' for s in splits) + ' |',
             '|---|' + '---:|' * (2 * len(splits) + 1)]
    tot = Counter()
    for c in classes:
        row = [counts.get((s, c), 0) for s in splits]
        n = sum(row)
        for s, v in zip(splits, row):
            tot[s] += v
        lines.append(f'| {c} | ' + ' | '.join(map(str, row)) + f' | {n} | ' +
                     ' | '.join(f'{100 * v / max(n, 1):.1f}' for v in row) + ' |')
    n = sum(tot.values())
    lines.append('| **total** | ' + ' | '.join(str(tot[s]) for s in splits) + f' | {n} | ' +
                 ' | '.join(f'{100 * tot[s] / max(n, 1):.1f}' for s in splits) + ' |')
    return lines


def montage(pairs, recs, path, title_fn, max_pairs=16, size=192):
    if not pairs:
        return
    pairs = pairs[:max_pairs]
    canvas = Image.new('RGB', (2 * size, size * len(pairs)), 'white')
    for k, pr in enumerate(pairs):
        for col, idx in enumerate(pr[2:4]):
            try:
                im = Image.open(recs[idx]['src'])
                im.draft('RGB', (size, size))
                canvas.paste(im.convert('RGB').resize((size, size)), (col * size, k * size))
            except Exception:  # noqa: BLE001
                pass
    canvas.save(path, quality=85)
    with open(os.path.splitext(path)[0] + '.txt', 'w') as f:
        for pr in pairs:
            f.write(title_fn(pr) + '\n')


def export(recs, out_root, max_side):
    for r in recs:
        dst_dir = os.path.join(out_root, r['split'], r['cls'])
        os.makedirs(dst_dir, exist_ok=True)
        name = os.path.basename(r['src'])
        if r.get('rename'):
            name = r['rename']
        dst = os.path.join(dst_dir, name)
        if max_side:
            dst = os.path.splitext(dst)[0] + '.jpg'
            im = Image.open(r['src']).convert('RGB')
            if max(im.size) > max_side:
                s = max_side / max(im.size)
                im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
            im.save(dst, quality=95)
        else:
            shutil.copy2(r['src'], dst)
        r['out'] = os.path.relpath(dst, out_root)


# ----------------------------------------------------------------------------- main per dataset
def reason_key(reason):
    """removed.csv reason -> category used in the report and the paper table."""
    for key in ('roboflow copy', 'exact duplicate', 'conflicting labels', 'leakage', 'several classes', 'no boxes',
                'unreadable', 'class has', 'different labels', 'shooting session'):
        if key in reason:
            return key
    return reason


def prepare(name, recs, removed, a, report, resplit, n_files=None):
    n_files = n_files or len(recs)
    print(f"\n{'=' * 70}\n{name}: {len(recs)} images")
    report += [f'## {name}', '']
    ck = os.path.join(a.out, '_checks')
    os.makedirs(ck, exist_ok=True)

    feats = compute_features([r['src'] for r in recs], a.workers)
    bad = [i for i, f in enumerate(feats) if isinstance(f, str)]
    for i in bad:
        removed.append((recs[i]['rel'], recs[i]['split'] or '', recs[i]['cls'], f'unreadable: {feats[i]}'))
    keep = [i for i in range(len(recs)) if i not in set(bad)]
    recs, feats = [recs[i] for i in keep], [feats[i] for i in keep]
    for r, f in zip(recs, feats):
        r['md5'], r['size'] = f[0], f[3]

    dup, grp, dup_pairs, grp_pairs = find_duplicates(feats, [r['cls'] for r in recs], a.hash_dist,
                                                     a.dup_thr, a.group_thr)
    title = lambda p: f"crop-MAE={p[0]:.2f} hash-dist={p[1]} | {recs[p[2]]['rel']} [{recs[p[2]]['cls']}] | " \
                      f"{recs[p[3]]['rel']} [{recs[p[3]]['cls']}]"    # noqa: E731
    # visual checks: the least similar pairs of each kind (where a wrong decision would show)
    montage(dup_pairs[::-1], recs, os.path.join(ck, f'{name}_duplicates.jpg'), title)
    montage(grp_pairs[::-1], recs, os.path.join(ck, f'{name}_same_split_groups.jpg'), title)
    dup_groups, split_groups = defaultdict(list), defaultdict(list)
    for i in range(len(recs)):
        dup_groups[dup.find(i)].append(i)
        split_groups[grp.find(i)].append(i)
    dup_groups = [g for g in dup_groups.values() if len(g) > 1]
    n_exact = len(recs) - len({r['md5'] for r in recs})
    big = max((len(g) for g in split_groups.values()), default=1)
    print(f"  exact duplicate files: {n_exact} | near-duplicate pairs: {len(dup_pairs)} -> "
          f"{len(dup_groups)} duplicate groups ({sum(len(g) for g in dup_groups)} images) | "
          f"same-leaf pairs kept in one split: {len(grp_pairs)} (largest group {big} images)")

    # duplicates whose copies carry different labels -> label noise, removed
    drop = set()
    n_conflict = 0
    for g in dup_groups:
        if len({recs[i]['cls'] for i in g}) > 1:
            n_conflict += 1
            for i in g:
                drop.add(i)
                removed.append((recs[i]['rel'], recs[i]['split'] or '', recs[i]['cls'],
                                'duplicate group with conflicting labels: ' +
                                ' / '.join(sorted({recs[j]['cls'] for j in g}))))
    # byte-identical files (same label): keep one copy. Near duplicates (burst shots) are different photos and
    # are kept; their group goes to one split.
    by_md5 = defaultdict(list)
    for i, r in enumerate(recs):
        if i not in drop:
            by_md5[r['md5']].append(i)
    for idx in by_md5.values():
        idx = sorted(idx, key=lambda i: recs[i]['rel'])
        for i in idx[1:]:
            drop.add(i)
            removed.append((recs[i]['rel'], recs[i]['split'] or '', recs[i]['cls'],
                            f"exact duplicate of {recs[idx[0]]['rel']}"))
    # official split kept: a group spanning splits -> keep its test (then val) images, drop the others.
    # --split_mode official keeps them all (leaky by design, for the effect-of-leakage experiment).
    n_leak = 0
    if not resplit:
        order = {'test': 0, 'val': 1, 'train': 2}
        for g in split_groups.values():
            g = [i for i in g if i not in drop]
            sp = {recs[i]['split'] for i in g}
            if len(sp) > 1:
                n_leak += 1
                if a.split_mode == 'official':
                    continue
                best = min(sp, key=order.get)
                for i in g:
                    if recs[i]['split'] != best:
                        drop.add(i)
                        removed.append((recs[i]['rel'], recs[i]['split'], recs[i]['cls'],
                                        f'same leaf as a {best} image (leakage)'))
    n_exact_removed = sum(len(v) - 1 for v in by_md5.values())
    print(f"  removed: {n_conflict} duplicate groups with conflicting labels, {n_exact_removed} exact duplicate files"
          + ('' if resplit else f", {n_leak} groups spanning the official splits"))

    for i, r in enumerate(recs):
        r['group'] = grp.find(i)
    groups = {k: v for k, v in split_groups.items()}
    accepted = dup_pairs
    recs = [r for i, r in enumerate(recs) if i not in drop]

    # small classes
    cnt = Counter(r['cls'] for r in recs)
    small = sorted(c for c, n in cnt.items() if n < a.min_class_size)
    if small:
        print(f"  classes with < {a.min_class_size} images removed: {[(c, cnt[c]) for c in small]}")
        for r in recs:
            if r['cls'] in small:
                removed.append((r['rel'], r['split'] or '', r['cls'], f'class has < {a.min_class_size} images'))
        recs = [r for r in recs if r['cls'] not in small]

    # split. grouped: duplicate groups are the unit (leakage-free). random: every image is its own unit, so copies
    # of one leaf can land in train and test (leaky by design, same images as grouped, for the leakage experiment).
    # shooting sessions (field photos with frame numbers): consecutive frames are often the same leaf re-framed, which
    # the leaf-crop test cannot detect -> whole sessions per split (session_split)
    sessions_on = name in CAPTURE_ORDER_DATASETS and not a.no_capture_order
    capture = resplit and a.split_mode == 'grouped' and sessions_on
    cuts, n_adj_official = [], None
    if name in CAPTURE_ORDER_DATASETS and all(r['split'] for r in recs):     # official split, before re-splitting
        fr = sorted((r['cls'], frame_no(r['rel']), r['split']) for r in recs)
        n_adj_official = sum(1 for x, y in zip(fr, fr[1:])
                             if x[0] == y[0] and y[1] - x[1] <= a.burst_gap and x[2] != y[2])
    if sessions_on:
        members = defaultdict(list)
        for r in recs:
            members[r['group']].append(r)
        units = [(u, rs[0]['cls'], min(frame_no(r['rel']) for r in rs), max(frame_no(r['rel']) for r in rs), len(rs))
                 for u, rs in members.items()]
        assign, cuts = session_split(units, a.ratios, a.session_gap, SESSION_CUTS.get(name))
    if sessions_on and not capture:
        # leaky variants (random / official): the same images as the session split, so classes it excludes go here too
        excl = {c for c, t, ul, _ in cuts if ul is None and t.startswith('EXCLUDED')}
        for r in recs:
            if r['cls'] in excl:
                removed.append((r['rel'], r['split'] or '', r['cls'], 'class cannot be split by shooting session'))
        recs = [r for r in recs if r['cls'] not in excl]
        print(f"  classes excluded as in the session split: {sorted(excl) or 'none'}")
        cuts = []
    if capture:
        for r in recs:
            k = assign[r['group']]
            r['split'] = None if k is None else ('train', 'val', 'test')[k]
            if k is None:
                removed.append((r['rel'], '', r['cls'], 'class cannot be split by shooting session'))
        print('  shooting sessions:\n' + '\n'.join(f'    {c}: {t}' for c, t, ul, _ in cuts if ul is None))
        # visual check: the closest frames on the two sides of every boundary between sessions of different splits
        idx = {id(r): i for i, r in enumerate(recs)}
        near = lambda u, last: max(members[u], key=lambda r: frame_no(r['rel']) * (1 if last else -1))  # noqa: E731
        cut_pairs = [(0, 0, idx[id(near(ul, True))], idx[id(near(ur, False))], c, t)
                     for c, t, ul, ur in cuts if ul is not None]
        montage(cut_pairs, recs, os.path.join(ck, f'{name}_split_cuts.jpg'),
                lambda p: f"{p[4]}: {p[5]} | {recs[p[2]]['rel']} [{recs[p[2]]['split']}] | "
                          f"{recs[p[3]]['rel']} [{recs[p[3]]['split']}]", max_pairs=40)
        recs = [r for r in recs if r['split'] is not None]
        ratios = '/'.join(f'{x:g}' for x in a.ratios)
        how = (f"{ratios} split per class by shooting session (new session after a pause of >= {a.session_gap} "
               f"frame numbers), whole sessions per split; duplicate groups kept together")
    elif resplit:
        unit = (lambda r: r['rel']) if a.split_mode == 'random' else (lambda r: f"{r['group']}")
        units = Counter((unit(r), r['cls']) for r in recs)
        assign = stratified_split([(u, c, n) for (u, c), n in units.items()], a.ratios, a.seed)
        for r in recs:
            r['split'] = ('train', 'val', 'test')[assign[unit(r)]]
        ratios = '/'.join(f'{x:g}' for x in a.ratios)
        how = (f"stratified {ratios} split, seed {a.seed}, image by image -- duplicate groups IGNORED (leaky by design)"
               if a.split_mode == 'random' else
               f"stratified {ratios} split, seed {a.seed}, duplicate groups kept together")
    elif a.split_mode == 'official':
        how = 'official split of the dataset, near-identical shots across splits KEPT (leaky by design)'
    else:
        how = 'official split of the dataset (kept, leaked training images removed)'

    # leakage check of the final split: test / val images that have a same-leaf image in train
    group_splits = defaultdict(set)
    for r in recs:
        group_splits[r['group']].add(r['split'])
    leak = {s: sum(1 for r in recs if r['split'] == s and 'train' in group_splits[r['group']]) for s in ('val', 'test')}
    n_split = Counter(r['split'] for r in recs)
    print(f"  leakage check: {leak['test']}/{n_split['test']} test and {leak['val']}/{n_split['val']} val images have "
          f"a same-leaf image in train")
    # capture-order check: consecutive frames (same class, frame gap <= burst_gap) placed in different splits
    n_adj, n_adj_train, min_dist = None, None, None
    if name in CAPTURE_ORDER_DATASETS:
        fr = sorted((r['cls'], frame_no(r['rel']), r['split']) for r in recs)
        pairs = [(x, y) for x, y in zip(fr, fr[1:]) if x[0] == y[0] and y[1] - x[1] <= a.burst_gap and x[2] != y[2]]
        n_adj = len(pairs)
        n_adj_train = sum(1 for x, y in pairs if 'train' in (x[2], y[2]))
        print(f"  capture-order check: {n_adj} pairs of consecutive frames (gap <= {a.burst_gap}) in different splits, "
              f"{n_adj_train} of them between train and val / test")
        last = defaultdict(dict)                   # per class: closest frames of train and val / test
        for cls, f, s in fr:
            s = s == 'train'
            o = last[cls].get(not s)               # latest earlier frame of the other side
            if o is not None and (min_dist is None or f - o < min_dist):
                min_dist = f - o
            last[cls][s] = f
        print(f"  capture-order check: closest frames of train and val / test are {min_dist} numbers apart")

    # unique output names (YOLO: base name without the Roboflow hash)
    for r in recs:
        if 'n_copies' in r:
            r['rename'] = re.sub(r'\.rf\.[0-9a-f]+(\.\w+)$', r'\1', os.path.basename(r['src']))
    classes = sorted({r['cls'] for r in recs})
    counts = Counter((r['split'], r['cls']) for r in recs)
    print(f"  split: {how}")
    print('\n'.join('  ' + line for line in table(counts, classes)))

    out_root = os.path.join(a.out, name)
    if not a.dry_run:
        if os.path.exists(out_root):
            shutil.rmtree(out_root)               # rebuilt from scratch: no images of an older split remain
        export(recs, out_root, a.max_side)
        with open(os.path.join(out_root, 'manifest.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['split', 'class', 'file', 'source', 'md5', 'width', 'height', 'dup_group'])
            for r in sorted(recs, key=lambda r: (r['split'], r['cls'], r['out'])):
                w.writerow([r['split'], r['cls'], r['out'], r['rel'], r['md5'], r['size'][0], r['size'][1],
                            r['group'] if len(groups[r['group']]) > 1 else ''])
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'removed.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'split', 'class', 'reason'])
        w.writerows(sorted(x for x in removed))

    reasons = Counter(reason_key(x[3]) for x in removed)
    n_kept_groups = [k for k in (set(g) - drop for g in split_groups.values()) if len(k) > 1]
    stats = {'n_files': n_files, 'n_final': len(recs), 'removed': dict(reasons), 'resplit': resplit,
             'split_mode': a.split_mode, 'test_with_train_duplicate': leak['test'],
             'val_with_train_duplicate': leak['val'], 'n_official_groups_across_splits': n_leak,
             'n_near_dup_groups': len(dup_groups), 'n_near_dup_images': sum(len(g) for g in dup_groups),
             'n_same_split_groups': len(n_kept_groups), 'n_same_split_images': sum(len(g) for g in n_kept_groups),
             'n_leak_groups_official': n_leak, 'small_classes': {c: cnt[c] for c in small},
             'capture_order': capture, 'adjacent_frames_across_splits': n_adj, 'official_adjacent': n_adj_official,
             'adjacent_frames_train_heldout': n_adj_train, 'min_frame_distance_train_heldout': min_dist,
             'sessions': [f'{c}: {t}' for c, t, ul, _ in cuts if ul is None],
             'excluded_classes': sorted({c for c, t, ul, _ in cuts if ul is None and t.startswith('EXCLUDED')}),
             'per_class': {c: [counts.get((s, c), 0) for s in ('train', 'val', 'test')] for c in classes}}
    report += [f'Split: {how}.', '',
               f'- exact duplicate files: {n_exact}; near-duplicate pairs (leaf-crop MAE <= {a.dup_thr}): '
               f'{len(accepted)}; duplicate groups: {len(dup_groups)}; same-label same-leaf pairs kept in one split '
               f'(MAE <= {a.group_thr}): {len(grp_pairs)}',
               f'- groups with conflicting labels (removed): {n_conflict}; groups spanning the official splits: {n_leak}',
               '- removed images by reason: ' + (', '.join(f'{k}: {v}' for k, v in reasons.items()) or 'none'),
               f'- classes removed (< {a.min_class_size} images): {small or "none"}']
    if n_adj is not None:
        report += [f'- consecutive frames (gap <= {a.burst_gap}) in different splits: {n_adj} '
                   f'({n_adj_train} between train and val / test); closest frames of train and val / test: '
                   f'{min_dist} numbers apart']
    report += [f'- sessions {x}' for x in stats['sessions']] + [''] + table(counts, classes) + ['']
    return {'classes': classes, 'n': len(recs), 'counts': {f'{s}/{c}': n for (s, c), n in counts.items()},
            'stats': stats}


DATASET_INFO = {   # name in the paper, citation key(s) of the dataset
    'TeaLeafBD': ('teaLeafBD', 'teaLeafBD'),
    'TeaSickness': ('Tea Sickness Dataset', 'kimutai2022tea'),
    'TeaDisease': ('Tea-Leaf Disease and Pest Detection Dataset', 'lyu2026tea,lv2026teadataset'),
}


def plural(n, word):
    return f'{n} {word}' + ('' if n == 1 else 's')


# Display names for the paper tables. Same list as data/class_names.py (copied: this script runs standalone).
CLASS_DISPLAY = {
    'algal_leaf_spot': 'Algal Leaf Spot', 'anthracnose': 'Anthracnose', 'bird_eye_spot': "Bird's Eye Spot",
    'black_rot': 'Black Rot', 'brown_blight': 'Brown Blight', 'gray_blight': 'Grey Blight',
    'green_mirid_bug': 'Green Mirid Bug', 'healthy': 'Healthy', 'helopeltis': r'\emph{Helopeltis}',
    'leaf_rust': 'Leaf Rust', 'red_leaf_spot': 'Red Leaf Spot', 'red_spider': 'Red Spider Mite',
    'tea_mosquito_bug': 'Tea Mosquito Bug', 'white_spot': 'White Spot',
}


def class_tex(c):
    return CLASS_DISPLAY.get(c, tex(c.replace('_', ' ')))


def cite(n):
    """Dataset name with its citation, for the table cells."""
    return f'{DATASET_INFO[n][0]}~\\citep{{{DATASET_INFO[n][1]}}}'


def tex(s):
    return str(s).replace('_', '\\_').replace('%', '\\%')


def write_paper_section(summary, a):
    """LaTeX paragraph + two tables with the numbers of THIS run -> <out>/paper_dataset_section.tex"""
    if not summary:
        return
    r = [f'{x:g}' for x in (100 * a.ratios[0], 100 * a.ratios[1], 100 * a.ratios[2])]
    L = ['% Generated by tools/prepare_data.py -- the numbers are those of this run; regenerate, do not edit by hand.',
         f'% command: python {" ".join(sys.argv)}', '',
         '\\subsection{Datasets and data preparation}\\label{sec:data}', '']
    names = list(summary)
    desc = ', '.join(f"{DATASET_INFO[n][0]}~\\citep{{{DATASET_INFO[n][1]}}} ({len(summary[n]['classes'])} classes)"
                     for n in names)
    L.append(f'We use {len(names)} public tea leaf disease datasets: {desc}. '
             'Below they are called teaLeafBD, the Tea Sickness Dataset and the TeaDisease dataset (the last after conversion to '
             'classification), with the short labels teaLeafBD, Tea Sickness and TeaDisease in figures and tables. '
             'Image collections often contain repeated photographs of the same leaf; when such photographs are in '
             'both the training and the test set, the test score also reflects recognition of leaves seen in '
             'training. To measure recognition of new leaves, we prepared every dataset and created new splits with '
             'one protocol before any experiment:')
    L += ['\\begin{enumerate}']
    if 'TeaDisease' in summary:
        s = summary['TeaDisease']['stats']
        rm = s['removed']
        small = [("the unspecified class ``disease''" if c == 'disease' else class_tex(c).lower()) + f' ({n})'
                 for c, n in s['small_classes'].items()]
        small = ', '.join(small[:-1]) + ' and ' + small[-1] if len(small) > 1 else ''.join(small)
        n_photos = s['n_files'] - rm.get('roboflow copy', 0)
        L.append(f"\\item \\emph{{Conversion of the detection dataset.}} The TeaDisease dataset is distributed "
                 f"as a YOLO object-detection export in which most photographs are stored several times as "
                 f"pixel-identical copies ({s['n_files']} files, but only {n_photos} distinct photographs, "
                 f"{s['n_files'] / max(n_photos, 1):.1f} files per photograph on average). "
                 f"We kept one copy per photograph and used the class of its bounding boxes as image label; "
                 + plural(rm.get('several classes', 0), 'photograph') + ' with boxes of several classes '
                 + ('was' if rm.get('several classes', 0) == 1 else 'were') + ' excluded'
                 + (f", as were classes with fewer than {a.min_class_size} photographs: {small}." if small else '.'))
    L.append('\\item \\emph{Exact duplicates.} Byte-identical files were detected by their MD5 checksum and only one '
             'copy was kept; copies with different labels were removed as label noise.')
    L.append('\\item \\emph{Near duplicates.} Most images show a single leaf, often photographed several times in a '
             'row. Whole-image perceptual hashes are dominated by the uniform background and match different leaves, '
             'so we compared the leaves themselves: the leaf was cropped from the background (pixels that differ from '
             'the median border colour), and two images were considered the same leaf when the mean absolute '
             f'difference of their mean-centred $48\\times48$ RGB crops was at most {a.dup_thr:g} (of 255); '
             f'images of the same class with a difference of up to {a.group_thr:g} were also grouped. '
             'Thresholds were set by visual '
             'inspection of image pairs. Near duplicates with different labels were removed; the others were kept, '
             'but every group was assigned to a single split, so no detected duplicate crosses the splits ('
             + '; '.join(f"{DATASET_INFO[n][0]}: {plural(summary[n]['stats']['n_same_split_groups'], 'group')} "
                         f"with {summary[n]['stats']['n_same_split_images']} images" for n in names) + ').')
    L.append(f'\\item \\emph{{Splitting.}} Each dataset was split into {r[0]}\\,\\% training, {r[1]}\\,\\% validation '
             f'and {r[2]}\\,\\% test images, stratified by class (seed {a.seed}), with duplicate groups as the unit of '
             'assignment' + ('. The split released with the detection dataset serves object detection, where '
                             'photographs of the same scene may be on both sides; for classification of new leaves '
                             'we split it anew (next item)'
                             if summary.get('TeaDisease', {}).get('stats', {}).get('resplit') else '') + '.')
    td = summary.get('TeaDisease', {}).get('stats', {})
    if td.get('capture_order'):
        excl = td.get('excluded_classes', [])
        n_excl = td['removed'].get('shooting session', 0)
        excl_txt = (' and '.join(class_tex(c) for c in excl) + f' ({n_excl} images) '
                    + ('was' if len(excl) == 1 else 'were') + ' excluded from this dataset because '
                    + ('almost all of its photographs come' if len(excl) == 1 else 'almost all of their photographs '
                                                                                   'come')
                    + ' from a single session, which cannot be divided without sharing leaves between the training '
                      'and the test set. ') if excl else ''
        L.append(f"\\item \\emph{{Shooting sessions.}} The photographs of the TeaDisease dataset were "
                 'close-ups of leaves on the plant whose file names carry a frame number; the documentation of the dataset does not describe their capture (Table~\\ref{tab:provenance}). Consecutive frames often show the same leaf '
                 'from slightly different positions (Fig.~\\ref{fig:bursts}), and a leaf sometimes reappears '
                 'later in the same session. The background changes between such frames, so neither the '
                 'checksum nor the leaf-crop comparison detects them; in the split released with the dataset, '
                 f"{td.get('official_adjacent', '--')} pairs of consecutive frames of one class are in different "
                 'sets. '
                 'We therefore grouped the images of each class into sessions, clusters inferred from the frame number of the '
                 f'file name (a new session starts after a gap of at least {a.session_gap} numbers), and '
                 'assigned whole sessions to the sets, choosing per '
                 'class the assignment closest to the target proportions. One session of the healthy class had to be '
                 'divided between the validation and the test set, at a change of leaf checked by eye; no session '
                 'is shared between the training set and the validation or test set. ' + excl_txt
                 + 'With few sessions per class, the proportions of some classes deviate from '
                 f'{r[0]}/{r[1]}/{r[2]}\\,\\% (Table~S1 in the Supplementary material).')
    L += ['\\end{enumerate}',
          'All models are trained with the same hyper-parameters; the checkpoint of each run is selected on the '
          'validation set, and the test set is used only once, for the final evaluation. '
          'Table~\\ref{tab:datasets} summarises the cleaning and Table~S1 in the Supplementary material the resulting splits. '
          'The split files are released with the code.', '']

    # overview table
    keys = [('roboflow copy', 'Repeated copies'), ('exact duplicate', 'Exact dupl.'),
            ('conflicting labels', 'Label conflicts'), ('several classes', 'Multi-class'), ('class has', 'Small classes'),
            ('shooting session', 'One session')]
    keys = [(k, h) for k, h in keys if any(summary[n]['stats']['removed'].get(k) for n in names)]
    L += ['\\begin{table}[t]', '\\centering',
          '\\caption{Datasets before and after cleaning (number of images).}', '\\label{tab:datasets}',
          '\\resizebox{\\linewidth}{!}{%',     # 10 columns: scale to the column / page width (needs graphicx)
          '\\begin{tabular}{p{4.2cm}' + 'r' * (4 + len(keys)) + '}', '\\toprule',
          'Dataset & Classes & Files & ' + ' & '.join(h for _, h in keys) + ' & Final & Train / Val / Test \\\\',
          '\\midrule']
    for n in names:
        s = summary[n]['stats']
        tvt = [sum(v[k] for v in s['per_class'].values()) for k in range(3)]
        other = sum(v for k, v in s['removed'].items() if k not in {kk for kk, _ in keys})
        L.append(f"{cite(n)} & {len(summary[n]['classes'])} & {s['n_files']} & "
                 + ' & '.join(str(s['removed'].get(k, 0)) for k, _ in keys)
                 + f" & {s['n_final']} & {tvt[0]} / {tvt[1]} / {tvt[2]} \\\\"
                 + (f' % other removed: {other}' if other else ''))
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}', '']

    # per-class table
    C = ['\\begin{table}[t]', '\\centering', '\\caption{Images per class and split after cleaning.}',
          '\\label{tab:class_counts}', '\\begin{tabular}{p{4.6cm}lrrrr}', '\\toprule',
          'Dataset & Class & Train & Val & Test & Total \\\\', '\\midrule']
    for k, n in enumerate(names):
        pc = summary[n]['stats']['per_class']
        for j, (c, v) in enumerate(pc.items()):
            C.append(f"{cite(n) if j == 0 else ''} & {class_tex(c)} & {v[0]} & {v[1]} & {v[2]} & {sum(v)} \\\\")
        if k < len(names) - 1:
            C.append('\\midrule')
    C += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    path = os.path.join(a.out, 'paper_dataset_section.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f"paper section (text + tables) -> {path}")
    with open(os.path.join(a.out, 'paper_class_counts.tex'), 'w') as f:      # Table S1 of the supplementary material
        f.write('\n'.join(C) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', default='datasets', help='folder with the three downloaded datasets')
    p.add_argument('--out', default='datasets_prepared')
    p.add_argument('--datasets', nargs='*', default=None, choices=['TeaLeafBD', 'TeaSickness', 'TeaDisease'],
                   help='default: all three')
    p.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=('TRAIN', 'VAL', 'TEST'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep_official_teadisease', action='store_true',
                   help='keep the official TeaDisease split (default: re-split 70/15/15 like the others; the official '
                        'split has near-identical shots in train and test and uneven class proportions)')
    p.add_argument('--min_class_size', type=int, default=30, help='classes with fewer unique images are removed')
    p.add_argument('--hash_dist', type=int, default=14, help='candidate pairs: max leaf-crop hash distance (of 64)')
    p.add_argument('--dup_thr', type=float, default=6.0,
                   help='duplicate: max mean abs. difference of the 48x48 leaf crops (0-255). Calibrated by eye: '
                        '<= 6 usually the same leaf, but a labelled check found exceptions')
    p.add_argument('--group_thr', type=float, default=9.0,
                   help='same label and leaf-crop difference <= this: kept in the same split (6-9 is often the '
                        'same leaf moved slightly, sometimes a similar leaf)')
    p.add_argument('--max_side', type=int, default=0, help='resize so the longer side is <= this (0 = copy files)')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--dry_run', action='store_true', help='report only, write no images')
    p.add_argument('--split_mode', default='grouped', choices=['grouped', 'random', 'official'],
                   help='grouped: leakage-free (default, the paper split). Effect-of-leakage experiment, same cleaned '
                        'images: random = image-by-image split ignoring duplicate groups; official = TeaDisease '
                        'official split with near-identical shots across splits kept (other datasets are skipped)')
    p.add_argument('--no_capture_order', action='store_true',
                   help='TeaDisease: stratified random split of the duplicate groups instead of the capture-order split')
    p.add_argument('--burst_gap', type=int, default=2,
                   help='TeaDisease: frames at most this far apart (camera frame number) belong to one burst')
    p.add_argument('--session_gap', type=int, default=12,
                   help='TeaDisease: a new shooting session starts after a pause of at least this many frame numbers')
    a = p.parse_args()
    assert abs(sum(a.ratios) - 1) < 1e-6, '--ratios must sum to 1'
    if a.split_mode != 'grouped' and a.out == 'datasets_prepared':
        sys.exit('leaky split modes must not overwrite the paper split: give another --out, e.g. datasets_leaky_random')
    os.makedirs(a.out, exist_ok=True)
    want = set(a.datasets or ['TeaLeafBD', 'TeaSickness', 'TeaDisease'])
    if a.split_mode == 'official':
        want &= {'TeaDisease'}                    # only TeaDisease has an official split

    report = ['# Dataset preparation report', '', f'{datetime.now():%Y-%m-%d %H:%M:%S}', '',
              '```', 'python ' + ' '.join(sys.argv), '```', '']
    summary = {}
    for folder, (name, cmap) in FOLDER_DATASETS.items():
        if name in want:
            removed = []
            recs = collect_folder_dataset(a.src, folder, cmap)
            summary[name] = prepare(name, recs, removed, a, report, resplit=True)
    folder, name, names = YOLO_DATASET
    if name in want:
        removed = []
        recs = collect_yolo_dataset(a.src, folder, names, removed)
        print(f"\n{name}: {len(recs)} unique photos kept from the YOLO export "
              f"({sum(1 for x in removed if x[3].startswith('roboflow copy'))} Roboflow copies dropped)")
        n_files = len(recs) + len(removed)            # all image files of the export
        resplit = not a.keep_official_teadisease and a.split_mode != 'official'
        summary[name] = prepare(name, recs, removed, a, report, resplit=resplit, n_files=n_files)
    if a.split_mode == 'grouped':
        write_paper_section(summary, a)           # the paper describes the leakage-free split only

    # class names across datasets (for cross-dataset tests)
    allc = sorted({c for s in summary.values() for c in s['classes']})
    report += ['## Classes across datasets', '', '| class | ' + ' | '.join(summary) + ' |',
               '|---|' + ':---:|' * len(summary)]
    for c in allc:
        report.append(f'| {c} | ' + ' | '.join('x' if c in s['classes'] else '' for s in summary.values()) + ' |')
    with open(os.path.join(a.out, 'prepare_report.md'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    with open(os.path.join(a.out, 'prepare_summary.json'), 'w') as f:
        json.dump({'args': vars(a), 'datasets': summary}, f, indent=1)
    print(f"\nreport -> {os.path.join(a.out, 'prepare_report.md')}")
    print(f"check the duplicate pairs by eye -> {os.path.join(a.out, '_checks')}/")


if __name__ == '__main__':
    main()
