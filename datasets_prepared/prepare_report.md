# Dataset preparation report

2026-10-22 00:20:57

```
python tools/prepare_data.py --src datasets --out datasets_prepared_v2
```

## TeaLeafBD

Split: stratified 0.7/0.15/0.15 split, seed 42, duplicate groups kept together.

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

Split: stratified 0.7/0.15/0.15 split, seed 42, duplicate groups kept together.

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

Split: 0.7/0.15/0.15 split per class by shooting session (new session after a pause of >= 12 frame numbers), whole sessions per split; duplicate groups kept together.

- exact duplicate files: 0; near-duplicate pairs (leaf-crop MAE <= 6.0): 541; duplicate groups: 183; same-label same-leaf pairs kept in one split (MAE <= 9.0): 681
- groups with conflicting labels (removed): 0; groups spanning the official splits: 0
- removed images by reason: roboflow copy: 6876, several classes: 1, class has: 3, shooting session: 96
- classes removed (< 30 images): ['brown_blight', 'disease']
- consecutive frames (gap <= 2) in different splits: 1 (0 between train and val / test); closest frames of train and val / test: 14 numbers apart
- sessions black_rot: 3 sessions -> 16 / 3 / 11: 773-775:val 886-901:train 985-995:test
- sessions healthy: 5 sessions -> 77 / 28 / 31: 511-584:train 1299-1329:test 1330-1357:val 1465-1465:train 3092-3093:train
- sessions leaf_rust: 9 sessions -> 466 / 97 / 98: 776-796:test 810-885:test 916-922:test 1586-2067:train 2410-2431:val 2751-2779:val 2807-2836:val 2944-2964:val 3041-3043:test
- sessions red_spider: 5 sessions -> 117 / 35 / 43: 177-226:train 442-476:val 585-638:train 730-772:test 797-809:train
- sessions tea_mosquito_bug: 14 sessions -> 1044 / 239 / 310: 151-159:val 244-260:val 295-441:train 477-510:train 639-729:train 860-866:train 923-982:train 996-997:val 1091-1298:train 1358-1578:val 2068-2404:train 2433-2747:test 2837-2943:train 2966-3040:train
- sessions white_spot: EXCLUDED, cannot be split by session (3 sessions, 96 images: 998-1090:train 1758-1761:test 2750-2750:val)

| class | train | val | test | total | train % | val % | test % |
|---|---:|---:|---:|---:|---:|---:|---:|
| black_rot | 16 | 3 | 11 | 30 | 53.3 | 10.0 | 36.7 |
| healthy | 77 | 28 | 31 | 136 | 56.6 | 20.6 | 22.8 |
| leaf_rust | 466 | 97 | 98 | 661 | 70.5 | 14.7 | 14.8 |
| red_spider | 117 | 35 | 43 | 195 | 60.0 | 17.9 | 22.1 |
| tea_mosquito_bug | 1044 | 239 | 310 | 1593 | 65.5 | 15.0 | 19.5 |
| **total** | 1720 | 402 | 493 | 2615 | 65.8 | 15.4 | 18.9 |

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
