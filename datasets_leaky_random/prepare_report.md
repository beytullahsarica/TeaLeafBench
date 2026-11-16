# Dataset preparation report

2026-10-22 00:23:11

```
python tools/prepare_data.py --src datasets --out datasets_leaky_random_v2 --split_mode random
```

## TeaLeafBD

Split: stratified 0.7/0.15/0.15 split, seed 42, image by image -- duplicate groups IGNORED (leaky by design).

- exact duplicate files: 1; near-duplicate pairs (leaf-crop MAE <= 6.0): 82; duplicate groups: 75; same-label same-leaf pairs kept in one split (MAE <= 9.0): 183
- groups with conflicting labels (removed): 1; groups spanning the official splits: 0
- removed images by reason: conflicting labels: 2, exact duplicate: 1
- classes removed (< 30 images): none

| class | train | val | test | total | train % | val % | test % |
|---|---:|---:|---:|---:|---:|---:|---:|
| algal_leaf_spot | 292 | 63 | 63 | 418 | 69.9 | 15.1 | 15.1 |
| brown_blight | 355 | 77 | 76 | 508 | 69.9 | 15.2 | 15.0 |
| gray_blight | 708 | 152 | 152 | 1012 | 70.0 | 15.0 | 15.0 |
| green_mirid_bug | 896 | 193 | 192 | 1281 | 69.9 | 15.1 | 15.0 |
| healthy | 654 | 140 | 140 | 934 | 70.0 | 15.0 | 15.0 |
| helopeltis | 425 | 91 | 91 | 607 | 70.0 | 15.0 | 15.0 |
| red_spider | 360 | 78 | 77 | 515 | 69.9 | 15.1 | 15.0 |
| **total** | 3690 | 794 | 791 | 5275 | 70.0 | 15.1 | 15.0 |

## TeaSickness

Split: stratified 0.7/0.15/0.15 split, seed 42, image by image -- duplicate groups IGNORED (leaky by design).

- exact duplicate files: 3; near-duplicate pairs (leaf-crop MAE <= 6.0): 30; duplicate groups: 10; same-label same-leaf pairs kept in one split (MAE <= 9.0): 23
- groups with conflicting labels (removed): 1; groups spanning the official splits: 0
- removed images by reason: conflicting labels: 2, exact duplicate: 2
- classes removed (< 30 images): none

| class | train | val | test | total | train % | val % | test % |
|---|---:|---:|---:|---:|---:|---:|---:|
| algal_leaf_spot | 79 | 17 | 17 | 113 | 69.9 | 15.0 | 15.0 |
| anthracnose | 69 | 15 | 15 | 99 | 69.7 | 15.2 | 15.2 |
| bird_eye_spot | 69 | 15 | 15 | 99 | 69.7 | 15.2 | 15.2 |
| brown_blight | 79 | 17 | 17 | 113 | 69.9 | 15.0 | 15.0 |
| gray_blight | 68 | 15 | 15 | 98 | 69.4 | 15.3 | 15.3 |
| healthy | 52 | 11 | 11 | 74 | 70.3 | 14.9 | 14.9 |
| red_leaf_spot | 99 | 22 | 22 | 143 | 69.2 | 15.4 | 15.4 |
| white_spot | 98 | 22 | 22 | 142 | 69.0 | 15.5 | 15.5 |
| **total** | 613 | 134 | 134 | 881 | 69.6 | 15.2 | 15.2 |

## TeaDisease

Split: stratified 0.7/0.15/0.15 split, seed 42, image by image -- duplicate groups IGNORED (leaky by design).

- exact duplicate files: 0; near-duplicate pairs (leaf-crop MAE <= 6.0): 541; duplicate groups: 183; same-label same-leaf pairs kept in one split (MAE <= 9.0): 681
- groups with conflicting labels (removed): 0; groups spanning the official splits: 0
- removed images by reason: roboflow copy: 6876, several classes: 1, class has: 3, shooting session: 96
- classes removed (< 30 images): ['brown_blight', 'disease']
- consecutive frames (gap <= 2) in different splits: 1192 (1073 between train and val / test); closest frames of train and val / test: 1 numbers apart

| class | train | val | test | total | train % | val % | test % |
|---|---:|---:|---:|---:|---:|---:|---:|
| black_rot | 20 | 5 | 5 | 30 | 66.7 | 16.7 | 16.7 |
| healthy | 94 | 21 | 21 | 136 | 69.1 | 15.4 | 15.4 |
| leaf_rust | 463 | 99 | 99 | 661 | 70.0 | 15.0 | 15.0 |
| red_spider | 136 | 30 | 29 | 195 | 69.7 | 15.4 | 14.9 |
| tea_mosquito_bug | 1115 | 239 | 239 | 1593 | 70.0 | 15.0 | 15.0 |
| **total** | 1828 | 394 | 393 | 2615 | 69.9 | 15.1 | 15.0 |

## Classes across datasets

| class | TeaLeafBD | TeaSickness | TeaDisease |
|---|:---:|:---:|:---:|
| algal_leaf_spot | x | x |  |
| anthracnose |  | x |  |
| bird_eye_spot |  | x |  |
| black_rot |  |  | x |
| brown_blight | x | x |  |
| gray_blight | x | x |  |
| green_mirid_bug | x |  |  |
| healthy | x | x | x |
| helopeltis | x |  |  |
| leaf_rust |  |  | x |
| red_leaf_spot |  | x |  |
| red_spider | x |  | x |
| tea_mosquito_bug |  |  | x |
| white_spot |  | x |  |
