#!/usr/bin/env bash

set -euo pipefail

script_dir="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
  pwd
)"
project_dir="$(
  cd -- "$script_dir/.."
  pwd
)"

cd "$project_dir"

experiment_mode="${1:-smoke}"
wandb_mode="${CS336_WANDB_MODE:-online}"
wandb_project="${CS336_WANDB_PROJECT:-cs336-assignment1}"

common_arguments=(
  --train-data data/tinystories_train.bin
  --val-data data/tinystories_validation.bin
  --wandb-mode "$wandb_mode"
  --wandb-project "$wandb_project"
  --vocab-size 10000
  --context-length 256
  --d-model 512
  --num-layers 4
  --num-heads 16
  --d-ff 1344
  --rope-theta 10000
  --batch-size 32
  --beta1 0.9
  --beta2 0.95
  --eps 1e-8
  --weight-decay 0.1
  --max-grad-norm 1.0
  --log-every 10
  --seed 336
  --device cuda
)

run_experiment() {
  local run_name="$1"
  local max_lr="$2"
  local min_lr="$3"
  local max_iters="$4"
  local warmup_iters="$5"
  local eval_every="$6"
  local eval_iters="$7"
  local save_every="$8"
  local run_dir="runs/tinystories/$run_name"

  mkdir -p "$run_dir/checkpoints"

  printf 'Starting %s\n' "$run_name"
  printf 'Logs: %s/train.log\n' "$run_dir"
  printf 'Checkpoints: %s/checkpoints\n' "$run_dir"

  uv run python scripts/train.py \
    "${common_arguments[@]}" \
    --checkpoint-dir "$run_dir/checkpoints" \
    --run-name "$run_name" \
    --max-iters "$max_iters" \
    --max-lr "$max_lr" \
    --min-lr "$min_lr" \
    --warmup-iters "$warmup_iters" \
    --cosine-cycle-iters "$max_iters" \
    --eval-every "$eval_every" \
    --eval-iters "$eval_iters" \
    --save-every "$save_every" \
    2>&1 | tee "$run_dir/train.log"
}

print_usage() {
  printf '%s\n' \
    "Usage:" \
    "  $0 smoke" \
    "  $0 sweep" \
    "  $0 full [MAX_LR] [MIN_LR]" \
    "" \
    "Environment variables:" \
    "  CS336_WANDB_MODE=offline|online|disabled" \
    "  CS336_WANDB_PROJECT=project-name"
}

case "$experiment_mode" in
  smoke)
    run_experiment \
      "ts-smoke-lr3e-4" \
      "3e-4" \
      "3e-5" \
      "100" \
      "10" \
      "25" \
      "10" \
      "100"
    ;;

  sweep)
    max_learning_rates=(
      "1e-4"
      "3e-4"
      "6e-4"
      "1e-3"
      "3e-3"
    )
    min_learning_rates=(
      "1e-5"
      "3e-5"
      "6e-5"
      "1e-4"
      "3e-4"
    )

    for index in "${!max_learning_rates[@]}"; do
      max_lr="${max_learning_rates[$index]}"
      min_lr="${min_learning_rates[$index]}"

      run_experiment \
        "ts-lr-${max_lr}-500steps" \
        "$max_lr" \
        "$min_lr" \
        "500" \
        "25" \
        "50" \
        "20" \
        "500"
    done
    ;;

  full)
    max_lr="${2:-3e-4}"
    min_lr="${3:-3e-5}"

    run_experiment \
      "ts-full-lr-${max_lr}-5000steps" \
      "$max_lr" \
      "$min_lr" \
      "5000" \
      "100" \
      "100" \
      "20" \
      "1000"
    ;;

  help|-h|--help)
    print_usage
    ;;

  *)
    printf 'Unknown mode: %s\n\n' "$experiment_mode" >&2
    print_usage >&2
    exit 2
    ;;
esac
