#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EPOCHS=10
BATCH_SIZE=256
HIDDEN_DIM=128
NUM_BLOCKS=2
SEED=42

DATASETS=("iris" "mnist" "fashion-mnist")
OPTIMIZERS=("sgd" "adam")
SCHEDULERS=("none" "step" "warmup" "cosine")

lr_for_optimizer() {
  case "$1" in
    sgd) echo "0.01" ;;
    adam) echo "0.001" ;;
    *) echo "unknown optimizer: $1" >&2; return 1 ;;
  esac
}

run_experiment() {
  local dataset="$1"
  local optimizer="$2"
  local scheduler="$3"
  local lr
  lr="$(lr_for_optimizer "${optimizer}")"

  local scheduler_args=("--scheduler" "${scheduler}")
  case "${scheduler}" in
    none)
      ;;
    step)
      scheduler_args+=("--step-size" "5" "--gamma" "0.5")
      ;;
    warmup)
      scheduler_args+=("--warmup-steps" "3" "--warmup-start-lr" "0.0")
      ;;
    cosine)
      scheduler_args+=("--cosine-first-cycle-steps" "5" "--cosine-min-lr" "0.0001")
      ;;
    *)
      echo "unknown scheduler: ${scheduler}" >&2
      return 1
      ;;
  esac

  echo "============================================================"
  echo "ResidualMLP experiment"
  echo "dataset=${dataset} optimizer=${optimizer} scheduler=${scheduler} hidden_dim=${HIDDEN_DIM} num_blocks=${NUM_BLOCKS} lr=${lr} epochs=${EPOCHS} batch_size=${BATCH_SIZE} seed=${SEED}"
  echo "============================================================"

  python -B apps/train_residual_mlp.py \
    --dataset "${dataset}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-blocks "${NUM_BLOCKS}" \
    --optimizer "${optimizer}" \
    --lr "${lr}" \
    "${scheduler_args[@]}" \
    --seed "${SEED}"
}

for dataset in "${DATASETS[@]}"; do
  for optimizer in "${OPTIMIZERS[@]}"; do
    for scheduler in "${SCHEDULERS[@]}"; do
      run_experiment "${dataset}" "${optimizer}" "${scheduler}"
    done
  done
done
