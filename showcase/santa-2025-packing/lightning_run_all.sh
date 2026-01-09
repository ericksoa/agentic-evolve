#!/bin/bash
# Lightning.ai Full Run Script
# Runs GPU SA (global search) + RL training

set -e

echo "========================================"
echo "Santa 2025 - Full Optimization Run"
echo "========================================"
echo

# Create output directory
OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"
echo

# ----------------------------------------
# Phase 1: GPU SA Global Search
# ----------------------------------------
echo "Phase 1: GPU SA Global Search"
echo "========================================"

# Small n values (n=1-20) - 500 chains, 10k iterations
echo "Running n=1-20..."
python python/gpu/lightning_run.py \
    --n-range 1-20 \
    --chains 500 \
    --iterations 10000 \
    --mode global \
    --output $OUTPUT_DIR/gpu_sa_n1-20.json

# Medium n values (n=21-50) - 300 chains, 15k iterations
echo "Running n=21-50..."
python python/gpu/lightning_run.py \
    --n-range 21-50 \
    --chains 300 \
    --iterations 15000 \
    --mode global \
    --output $OUTPUT_DIR/gpu_sa_n21-50.json

# Large n values (n=51-100) - 200 chains, 20k iterations
echo "Running n=51-100..."
python python/gpu/lightning_run.py \
    --n-range 51-100 \
    --chains 200 \
    --iterations 20000 \
    --mode global \
    --output $OUTPUT_DIR/gpu_sa_n51-100.json

echo
echo "GPU SA complete. Results in $OUTPUT_DIR/"
echo

# ----------------------------------------
# Phase 2: RL Training
# ----------------------------------------
echo "Phase 2: RL Training"
echo "========================================"

# Train on n=5 first (fastest)
echo "Training PPO for n=5..."
python python/rl/train_ppo.py \
    --n-trees 5 \
    --timesteps 500000 \
    --device cuda \
    --output-dir $OUTPUT_DIR/rl_n5

# Train on n=10
echo "Training PPO for n=10..."
python python/rl/train_ppo.py \
    --n-trees 10 \
    --timesteps 1000000 \
    --device cuda \
    --output-dir $OUTPUT_DIR/rl_n10

echo
echo "========================================"
echo "All runs complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "========================================"
