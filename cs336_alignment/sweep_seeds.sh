#!/usr/bin/env bash
# Random-seed sweep for the MaxRL variant (assignment 5.4,
# grpo_experiments_variants_on_policy): baseline=mean, advantage_normalizer=mean,
# loss_normalization=constant, on the zero-shot r1_zero prompt.
#
# 4 seeds as the assignment requires. Everything else -- including the learning
# rate -- is held at the values in train_grpo.py, so these runs stay comparable
# to the standard GRPO runs. Pass --lr to override if the tuned LR differs from
# the LEARNING_RATE default.
#
# Runs sequentially -- each job needs both GPUs (training on GPU 0, vLLM on
# GPU 1), so they cannot overlap.
#
# Usage:
#   ./sweep_seeds.sh                          # full sweep, default step count
#   ./sweep_seeds.sh --lr 3e-6                # extra args forwarded to every run
set -uo pipefail

SEEDS=(42 43 44 45)
# Keep in sync with VARIANT in train_grpo.py, so each variant's logs land in
# their own directory instead of overwriting the previous variant's.
VARIANT="maxrl"
LOG_DIR="logs/sweep_seeds_${VARIANT}"
mkdir -p "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
    log_file="$LOG_DIR/seed_${seed}.log"
    echo "=== Starting run: seed=${seed} (log: ${log_file}) ==="
    python train_grpo.py --seed "$seed" "$@" > "$log_file" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        echo "=== seed=${seed} FAILED with exit code ${status}; see ${log_file} ==="
    else
        echo "=== seed=${seed} finished ==="
    fi
done

echo "Sweep complete. Runs are in wandb project cs336-grpo as ${VARIANT}_lr=<lr>_seed=<seed>."
