# Dataset preparation report

2026-10-22 00:23:32

```
python tools/prepare_data.py --src datasets --out datasets_leaky_official_v2 --split_mode official
```

## TeaDisease

Split: official split of the dataset, near-identical shots across splits KEPT (leaky by design).

- exact duplicate files: 0; near-duplicate pairs (leaf-crop MAE <= 6.0): 541; duplicate groups: 183; same-label same-leaf pairs kept in one split (MAE <= 9.0): 681
- groups with conflicting labels (removed): 0; groups spanning the official splits: 158
- removed images by reason: roboflow copy: 6876, several classes: 1, class has: 3, shooting session: 96
- classes removed (< 30 images): ['brown_blight', 'disease']
- consecutive frames (gap <= 2) in different splits: 1012 (931 between train and val / test); closest frames of train and val / test: 1 numbers apart

| class | train | val | test | total | train % | val % | test % |
|---|---:|---:|---:|---:|---:|---:|---:|
| black_rot | 22 | 2 | 6 | 30 | 73.3 | 6.7 | 20.0 |
| healthy | 99 | 19 | 18 | 136 | 72.8 | 14.0 | 13.2 |
| leaf_rust | 486 | 66 | 109 | 661 | 73.5 | 10.0 | 16.5 |
| red_spider | 149 | 14 | 32 | 195 | 76.4 | 7.2 | 16.4 |
| tea_mosquito_bug | 1212 | 154 | 227 | 1593 | 76.1 | 9.7 | 14.2 |
| **total** | 1968 | 255 | 392 | 2615 | 75.3 | 9.8 | 15.0 |

## Classes across datasets

| class | TeaDisease |
|---|:---:|
| black_rot | x |
| healthy | x |
| leaf_rust | x |
| red_spider | x |
| tea_mosquito_bug | x |
