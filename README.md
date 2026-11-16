# TeaLeafBench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24223726-blue.svg)](https://doi.org/10.5281/zenodo.24223726)

Code and split files of the paper *"Split Effects, Background Cues and Cross-Dataset Transfer in Deep Tea Leaf Disease
Recognition: A Controlled Benchmark"* (B. Sarica, under review).

Twelve pretrained image classifiers are trained on three public tea leaf disease datasets with one recipe and three
seeds, on duplicate-aware splits (TeaDisease field photographs also split by shooting session), and compared on random
and released splits, across datasets and with background tests.

## Installation

```bash
conda create -n tealeaf python=3.10 -y && conda activate tealeaf
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install ultralytics==8.3.40      # only for YOLO11s-cls
python tools/check_models.py         # downloads and checks all pretrained weights
```

## Data

| Dataset | Source |
|---|---|
| teaLeafBD | Mendeley Data, doi:10.17632/744vznw5k2.4 |
| Tea Sickness Dataset | Mendeley Data, doi:10.17632/j32xdt2ff5 |
| TeaDisease | figshare, doi:10.6084/m9.figshare.32253357 |

The images are not redistributed. Download them into `datasets/` (`teaLeafBD/`, `tea sickness dataset/`,
`tea_disease_dataset/`) and prepare them once:

```bash
python tools/prepare_data.py --src datasets --out datasets_prepared                                # main splits
python tools/prepare_data.py --src datasets --out datasets_leaky_random   --split_mode random      # random split
python tools/prepare_data.py --src datasets --out datasets_leaky_official --split_mode official    # released split (TeaDisease only)
```

The `manifest.csv` files in this repository list every image with its split and MD5 checksum, so the splits of the
paper can be checked. The splits of the sensitivity analyses are in `splits_review/`.

## Training and evaluation

```bash
python train.py --dataset TeaLeafBD --model convnext_tiny --seed 0 --epochs 50   # one run -> results/<run>/
python train.py --eval_dir results/<run>                                    # re-evaluate a trained model
python train.py --eval_dir results/<run> --test_dataset TeaSickness         # cross-dataset test, shared classes
python train.py --dataset TeaDisease --model swin_t --seed 0 --epochs 50 --split_file splits_review/TeaDisease_p1.json
```

Models: `resnet50`, `efficientnetv2_s`, `convnext_tiny`, `mobilenetv4_m`, `efficientvit_b1`, `vit_b16`, `swin_t`,
`maxvit_t`, `mambaout_t`, `dinov2_s`, `segformer_b2`, `yolo11s_cls`.

All runs of the main benchmark (12 models x 3 datasets x 3 seeds):

```bash
SEEDS="0 1 2" bash scripts/run_benchmark.sh TeaLeafBD TeaSickness TeaDisease
SEEDS="0 1 2" TAG=leaky-random DATA_ROOT=datasets_leaky_random bash scripts/run_benchmark.sh TeaDisease
```

## Analyses, tables and figures

| Script | Purpose |
|---|---|
| `tools/make_review_splits.py`, `tools/make_cap_split.py` | split files in `splits_review/` |
| `tools/background_test.py`, `tools/review_interventions.py`, `tools/check_masks.py` | background and cross-dataset interventions |
| `tools/make_planted.py` | planted positive control |
| `tools/review_field.py` | session analyses of the field data |
| `tools/make_paper_tables.py`, `tools/make_review_tables.py`, `tools/make_paper_figures.py`, `tools/make_burst_figure.py` | tables and figures |

Each script's header gives its commands.

## Trained models

The 309 trained models (about 30 GB) are archived on Zenodo ([doi:10.5281/zenodo.24223726](https://doi.org/10.5281/zenodo.24223726)), with `MANIFEST.csv` listing each run's
test macro-F1 and the SHA-256 of its weights. Re-evaluate a model with `python train.py --eval_dir <run folder>`.

## Citation

The BibTeX entry will be added when the paper is published.

## License

Code: MIT. The three datasets are published by their authors under CC BY 4.0; please cite them when you use them.
