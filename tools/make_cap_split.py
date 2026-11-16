"""
TeaDisease session split with the three classes shared with teaLeafBD capped at the counts of the reduced teaLeafBD
training set (splits_review/TeaLeafBD_eq.json), so that both sides of the paper-versus-field comparison have the same
number of training images per scored class. Images are removed at random within the training sessions.

  python tools/make_cap_split.py    -> splits_review/TeaDisease_cap.json
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from make_review_splits import OUT, SHARED, read_manifest, subset_train, to_split  # noqa: E402


def main():
    if os.path.exists(os.path.join(OUT, 'TeaDisease_cap.json')):       # runs may already have used this split
        print('TeaDisease_cap.json exists, kept', flush=True)
        return
    with open(os.path.join(OUT, 'TeaLeafBD_eq.json')) as f:
        eq = Counter(c for _, c in json.load(f)['train'])
    td = read_manifest('TeaDisease')
    classes = sorted({r['class'] for r in td})
    n = Counter(r['class'] for r in td if r['split'] == 'train')
    counts = {c: eq[SHARED[c]] if c in SHARED else n[c] for c in classes}
    parts = subset_train(td, classes, counts, seed=0)
    with open(os.path.join(OUT, 'TeaDisease_cap.json'), 'w') as f:
        json.dump(to_split(classes, parts), f)
    print('train per class', dict(sorted(Counter(r['class'] for r in parts['train']).items())), 'val', len(parts['val']),
          'test', len(parts['test']), flush=True)


if __name__ == '__main__':
    main()
