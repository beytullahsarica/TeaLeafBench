"""
Tea leaf disease classification datasets. Two on-disk layouts are accepted under <data_root>/<dataset>/:

  1) pre-split (e.g. Roboflow exports):  {train, valid|val, test}/<class>/*.jpg
  2) class folders only:                 <class>/*.jpg   (images may be nested deeper inside a class folder)
     -> needs a split file: splits/<dataset>.json, or --split_file

A split file stores paths relative to the dataset folder, so the exact split can be shipped with the
repository and every model is trained / tested on the same images. If a split file exists it always wins
over the folders on disk.
"""
import csv
import hashlib
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(PROJECT_ROOT, 'splits')
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
SPLIT_DIRS = {'train': ('train',), 'val': ('val', 'valid', 'validation'), 'test': ('test',)}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def is_image(f):
    return f.lower().endswith(IMG_EXT) and not f.startswith('.')


def list_class_folder(root, classes=None):
    """[(path relative to root, class name)] for root/<class>/**/image. classes=None: every non-empty subfolder."""
    if classes is None:
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.'))
    samples = []
    for c in classes:
        cdir = os.path.join(root, c)
        if not os.path.isdir(cdir):
            continue
        for dp, _, fs in os.walk(cdir):
            samples += [(os.path.relpath(os.path.join(dp, f), root), c) for f in sorted(fs) if is_image(f)]
    return sorted(samples)


def find_split_dirs(root):
    """{'train': 'train', 'val': 'valid', 'test': 'test'} for the split folders that exist on disk."""
    found = {}
    for split, names in SPLIT_DIRS.items():
        for n in names:
            if os.path.isdir(os.path.join(root, n)):
                found[split] = n
                break
    return found


def looks_like_detection(root):
    """YOLO / COCO style export (images/ + labels/ or *.json annotations) instead of class folders."""
    for d in [root] + [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]:
        if os.path.isdir(os.path.join(d, 'labels')) or os.path.exists(os.path.join(d, '_annotations.coco.json')):
            return True
    return os.path.exists(os.path.join(root, 'data.yaml'))


def stratified_split(samples, ratios, seed=42, groups=None):
    """
    Split [(path, class)] into len(ratios) parts with the same class proportions.
    groups: optional {path: group id}; all images of one group (e.g. exact / near duplicates) go to the
    same part, so a duplicate can never end up in train and test at the same time.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for path, c in samples:
        by_class[c].append(path)
    parts = [[] for _ in ratios]
    for c in sorted(by_class):
        # unit of splitting = group of identical images
        units = defaultdict(list)
        for p in sorted(by_class[c]):
            units[groups.get(p, p) if groups else p].append(p)
        units = [units[k] for k in sorted(units)]
        rng.shuffle(units)
        n = sum(len(u) for u in units)
        targets = [r * n for r in ratios]
        counts = [0] * len(ratios)
        for u in units:
            # put the unit where it is most needed (largest remaining deficit relative to the target)
            k = max(range(len(ratios)), key=lambda i: (targets[i] - counts[i]) / max(targets[i], 1e-9))
            parts[k] += [(p, c) for p in u]
            counts[k] += len(u)
    return [sorted(p) for p in parts]


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_split(name, data_root, split_file=None, val_ratio=0.15, split_seed=42):
    """Returns root, classes, {'train'|'val'|'test': [(relpath, class)]}, info."""
    root = os.path.join(data_root, name)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset folder not found: {root}")
    sf = split_file or os.path.join(SPLITS_DIR, f'{name}.json')
    if os.path.exists(sf):
        with open(sf) as f:
            s = json.load(f)
        splits = {k: [tuple(x) for x in s[k]] for k in ('train', 'val', 'test')}
        info = {'source': 'split_file', 'split_file': os.path.relpath(sf, PROJECT_ROOT),
                **{k: s[k] for k in ('seed', 'ratios', 'dedup') if k in s}}
        return root, s['classes'], splits, info

    dirs = find_split_dirs(root)
    if 'train' not in dirs or 'test' not in dirs:
        hint = (' It looks like a DETECTION export (labels/ or COCO json), not class folders.'
                if looks_like_detection(root) else '')
        raise FileNotFoundError(
            f"{root} has no train/ + test/ folders and there is no split file {sf}.{hint}\n"
            f"Prepare the data first:  python tools/prepare_data.py --src datasets --out datasets_prepared")
    classes = sorted(set(c for _, c in list_class_folder(os.path.join(root, dirs['train']))))
    splits = {}
    for split, d in dirs.items():
        splits[split] = [(os.path.join(d, p), c)
                         for p, c in list_class_folder(os.path.join(root, d), classes)]
    info = {'source': 'folders', 'folders': dirs}
    if 'val' not in splits:
        splits['train'], splits['val'] = stratified_split(splits['train'], [1 - val_ratio, val_ratio], split_seed)
        info.update(val_from_train=True, val_ratio=val_ratio, split_seed=split_seed)
        print(f"[DATA] {name}: no val folder -> stratified {val_ratio:.0%} of train held out (seed {split_seed})")
    extra = sorted(set(c for _, c in splits['test']) - set(classes))
    if extra:
        print(f"[DATA][WARN] {name}: test classes not in train are ignored: {extra}")
    return root, classes, splits, info


def check_leakage(root, splits):
    """Exact duplicates (same bytes) of a test image in train/val -> list of offending paths."""
    test_hash = {file_md5(os.path.join(root, p)) for p, _ in splits['test']}
    return {s: [p for p, _ in splits[s] if file_md5(os.path.join(root, p)) in test_hash] for s in ('train', 'val')}


def build_transforms(size, mean=IMAGENET_MEAN, std=IMAGENET_STD, aug='standard'):
    """
    Train: random-resized crop (most of the leaf stays visible: lesions can sit at the leaf margin), flips,
    small rotations and mild colour jitter (hue kept small: colour is a disease cue). 'randaug' adds RandAugment.
    Eval: whole image resized to size x size (no centre crop, which could cut lesions off).
    """
    train = [T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
             T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20)]
    if aug == 'randaug':
        train.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == 'standard':
        train.append(T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02))
    train += [T.ToTensor(), T.Normalize(mean, std)]
    if aug == 'randaug':
        train.append(T.RandomErasing(p=0.25))
    evaluate = [T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(train), T.Compose(evaluate)


class LeafDataset(Dataset):
    def __init__(self, root, samples, classes, transform):
        self.root, self.samples, self.classes, self.transform = root, list(samples), list(classes), transform
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(p, c) for p, c in self.samples if c in self.class_to_idx]
        self.targets = [self.class_to_idx[c] for _, c in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c = self.samples[idx]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        return self.transform(img), self.class_to_idx[c]


def leaf_groups(ds):
    """Same-leaf group of every image (dup_group of manifest.csv, else the file itself), so that figures show each
    leaf only once (near-duplicates of one leaf stay together in one split and can both be in the test set)."""
    group = {}
    path = os.path.join(ds.root, 'manifest.csv')
    if os.path.exists(path):
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if r.get('dup_group', '').strip():
                    group[os.path.normpath(r['file'])] = r['dup_group'].strip()
    return [group.get(os.path.normpath(p), p) for p, _ in ds.samples]


def build_datasets(name, data_root, img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, aug='standard',
                   split_file=None, val_ratio=0.15, split_seed=42, check_dups=True):
    """Returns train_ds, val_ds, test_ds, info."""
    root, classes, splits, info = load_split(name, data_root, split_file, val_ratio, split_seed)
    if check_dups:
        leak = check_leakage(root, splits)
        n_leak = sum(len(v) for v in leak.values())
        info['n_test_duplicates_in_train_val'] = n_leak
        if n_leak:
            raise ValueError(f"[DATA] {name}: {len(leak['train'])} train / {len(leak['val'])} val images are byte-"
                             f"identical to test images (e.g. {(leak['train'] + leak['val'])[:3]}). Test leakage: "
                             f"re-split with tools/prepare_data.py (de-duplicates), or pass --allow_duplicates.")
    tr_tf, ev_tf = build_transforms(img_size, mean, std, aug)
    train_ds = LeafDataset(root, splits['train'], classes, tr_tf)
    val_ds = LeafDataset(root, splits['val'], classes, ev_tf)
    test_ds = LeafDataset(root, splits['test'], classes, ev_tf)
    info.update(root=root, classes=classes, n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds),
                class_counts={s: {c: sum(1 for _, x in d.samples if x == c) for c in classes}
                              for s, d in (('train', train_ds), ('val', val_ds), ('test', test_ds))})
    return train_ds, val_ds, test_ds, info


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
