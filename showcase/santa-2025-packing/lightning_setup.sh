#!/bin/bash
# Lightning.ai Setup Script for Santa 2025 Tree Packing
# Run this after creating a new L40S studio

set -e  # Exit on error

echo "========================================"
echo "Santa 2025 Tree Packing - Lightning.ai Setup"
echo "========================================"
echo

# Check GPU
echo "Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"
echo

# Install dependencies
echo "Installing dependencies..."
pip install -q stable-baselines3[extra] gymnasium tensorboard tqdm
echo "Dependencies installed."
echo

# Verify installation
echo "Verifying installation..."
python -c "from stable_baselines3 import PPO; print('stable-baselines3: OK')"
python -c "import gymnasium; print('gymnasium: OK')"
echo

echo "========================================"
echo "Setup complete! Ready to run:"
echo ""
echo "GPU SA (Global Search):"
echo "  python python/gpu/lightning_run.py --n-range 1-50 --chains 500 --iterations 10000 --mode global"
echo ""
echo "GPU SA (Refine Mode):"
echo "  python python/gpu/lightning_run.py --n-range 1-50 --chains 500 --iterations 5000 --mode refine"
echo ""
echo "RL Training:"
echo "  python python/rl/train_ppo.py --n-trees 5 --timesteps 500000 --device cuda"
echo "========================================"
