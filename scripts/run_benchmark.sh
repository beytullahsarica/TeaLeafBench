#!/bin/bash
# Runs of the paper, sequentially on one machine with a GPU: every model x every given dataset x every seed.
# Interrupted runs continue from their last checkpoint (--auto_resume); finished runs skip
# training and only repeat the (short) test evaluation.
#   bash scripts/run_benchmark.sh TeaLeafBD TeaSickness TeaDisease                 # 12 models, seed 0
#   SEEDS="1 2" MODELS="convnext_tiny maxvit_t swin_t" bash scripts/run_benchmark.sh TeaLeafBD
#   TAG=leaky-random DATA_ROOT=datasets_leaky_random bash scripts/run_benchmark.sh TeaDisease
set -e
cd "$(dirname "$0")/.."

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then echo "usage: $0 <dataset> [<dataset> ...]"; exit 1; fi
MODELS=${MODELS:-$(python -c "from models import DEFAULT_BENCHMARK; print(' '.join(DEFAULT_BENCHMARK))")}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-50}
IMG=${IMG:-224}
TAG=${TAG:-base}
EXTRA=()
[ -n "${DATA_ROOT:-}" ] && EXTRA=(--data_root "$DATA_ROOT")

for ds in "${DATASETS[@]}"; do
  for m in $MODELS; do
    for s in $SEEDS; do
      python train.py --dataset "$ds" --model "$m" --seed "$s" --epochs "$EPOCHS" --img_size "$IMG" \
          --batch_size 32 --tag "$TAG" "${EXTRA[@]}" --auto_resume
    done
  done
done
