#!/usr/bin/env bash
# Learning-rate sweep for GRPO on GSM8K.
#
# Runs one training job per LR, sequentially -- each job needs both GPUs
# (training on GPU 0, vLLM on GPU 1), so they cannot overlap.
#
# Usage:
#   ./sweep_lr.sh                  # full sweep, default step count
#   ./sweep_lr.sh --num-rollout-steps 200   # extra args forwarded to every run
set -uo pipefail

LRS=(1e-6 3e-6 1e-5 3e-5 1e-4)
LOG_DIR="logs/sweep_lr"
mkdir -p "$LOG_DIR"

for lr in "${LRS[@]}"; do
    log_file="$LOG_DIR/lr_${lr}.log"
    echo "=== Starting run: lr=${lr} (log: ${log_file}) ==="
    python train_grpo.py --lr "$lr" "$@" > "$log_file" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        echo "=== lr=${lr} FAILED with exit code ${status}; see ${log_file} ==="
    else
        echo "=== lr=${lr} finished ==="
    fi
done

echo "Sweep complete. Runs are in wandb project cs336-grpo as grpo_lr=<lr>."
