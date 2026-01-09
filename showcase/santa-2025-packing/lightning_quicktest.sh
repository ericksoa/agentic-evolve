#!/bin/bash
# Quick test to verify lightning.ai setup works
# Takes ~5 minutes

set -e

echo "========================================"
echo "Quick Test - Verify Setup"
echo "========================================"
echo

# Test GPU SA (small run)
echo "Testing GPU SA (n=5, 50 chains, 1000 iterations)..."
python python/gpu/lightning_run.py \
    --n-range 5-5 \
    --chains 50 \
    --iterations 1000 \
    --mode global \
    --output quicktest_gpu_sa.json

echo
cat quicktest_gpu_sa.json
echo

# Test RL training (very short)
echo "Testing RL training (n=5, 10000 timesteps)..."
python python/rl/train_ppo.py \
    --n-trees 5 \
    --timesteps 10000 \
    --device cuda \
    --output-dir quicktest_rl

echo
echo "========================================"
echo "Quick test passed! System is ready."
echo "========================================"
