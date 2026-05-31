#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EPOCHS=10
BATCH_SIZE=64
OPTIMIZER="adam"
LR=0.001
WEIGHT_DECAY=0.0
SEED=42
DEVICE="${DEVICE:-cpu_numpy}"

SCHEDULERS=("none" "step" "warmup" "cosine")

run_experiment() {
  local scheduler="$1"

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
      scheduler_args+=("--cosine-first-cycle-steps" "10" "--cosine-min-lr" "0.0001")
      ;;
    *)
      echo "unknown scheduler: ${scheduler}" >&2
      return 1
      ;;
  esac

  echo "============================================================"
  echo "ResNet9 experiment"
  echo "dataset=cifar10 optimizer=${OPTIMIZER} scheduler=${scheduler} augment=true lr=${LR} weight_decay=${WEIGHT_DECAY} epochs=${EPOCHS} batch_size=${BATCH_SIZE} seed=${SEED} device=${DEVICE}"
  echo "============================================================"

  python -B apps/train_resnet9.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --optimizer "${OPTIMIZER}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --device "${DEVICE}" \
    --augment \
    "${scheduler_args[@]}" \
    --seed "${SEED}"
}

for scheduler in "${SCHEDULERS[@]}"; do
  run_experiment "${scheduler}"
done
