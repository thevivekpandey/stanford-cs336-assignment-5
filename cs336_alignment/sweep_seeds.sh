#!/usr/bin/env bash
# Random-seed sweep for GRPO on GSM8K with the question_only prompt.
#
# LR is fixed at 1e-5 (LEARNING_RATE in train_grpo_other_prompts.py); the only
# thing varying across runs is the seed.
#
# Runs sequentially -- each job needs both GPUs (training on GPU 0, vLLM on
# GPU 1), so they cannot overlap.
#
# Usage:
#   ./sweep_seeds.sh                          # full sweep, default step count
#   ./sweep_seeds.sh --num-rollout-steps 50   # extra args forwarded to every run
set -uo pipefail

SEEDS=(42 43 44)
LOG_DIR="logs/sweep_seeds_question_only"
mkdir -p "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
    log_file="$LOG_DIR/seed_${seed}.log"
    echo "=== Starting run: seed=${seed} (log: ${log_file}) ==="
    python train_grpo_other_prompts.py --seed "$seed" "$@" > "$log_file" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        echo "=== seed=${seed} FAILED with exit code ${status}; see ${log_file} ==="
    else
        echo "=== seed=${seed} finished ==="
    fi
done

echo "Sweep complete. Runs are in wandb project cs336-grpo as grpo_question_only_seed=<seed>."
