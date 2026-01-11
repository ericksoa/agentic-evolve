#!/bin/bash
# Lightning.ai Setup Script for Chess LLM Training

set -e

echo "=============================================="
echo "Chess LLM Training Setup - Lightning.ai"
echo "=============================================="

# Create directories
mkdir -p ~/chess-training/{data,models,logs,results}

# Check GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: No GPU detected. Training will be slow."
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip

# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers ecosystem
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install accelerate>=0.25.0
pip install peft>=0.7.0
pip install trl>=0.7.0

# Quantization
pip install bitsandbytes>=0.41.0

# Chess
pip install python-chess>=1.10.0

# Evaluation & logging
pip install tensorboard
pip install wandb

# Try to install Flash Attention (may fail on some GPUs)
echo ""
echo "Attempting Flash Attention installation..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention not available (OK for L4/A10G)"

# Install Stockfish
echo ""
echo "Installing Stockfish..."
if ! command -v stockfish &> /dev/null; then
    # Try apt first (if available)
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y stockfish || true
    fi

    # If still not installed, download binary
    if ! command -v stockfish &> /dev/null; then
        echo "Downloading Stockfish binary..."
        cd /tmp
        wget -q https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar
        tar -xf stockfish-ubuntu-x86-64-avx2.tar
        sudo mv stockfish/stockfish-ubuntu-x86-64-avx2 /usr/local/bin/stockfish
        sudo chmod +x /usr/local/bin/stockfish
        rm -rf stockfish stockfish-ubuntu-x86-64-avx2.tar
        cd -
    fi
fi

# Verify Stockfish
if command -v stockfish &> /dev/null; then
    echo "Stockfish installed: $(which stockfish)"
else
    echo "WARNING: Stockfish not installed. Evaluation will be limited."
fi

# Download base model (cache it)
echo ""
echo "Caching base model (Qwen2.5-3B)..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
print('Downloading model config...')
# Just download config, not full weights (save time)
from transformers import AutoConfig
AutoConfig.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
print('Model cached.')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
echo "  ~/chess-training/data/    - Training data"
echo "  ~/chess-training/models/  - Trained models"
echo "  ~/chess-training/logs/    - Training logs"
echo "  ~/chess-training/results/ - Evaluation results"
echo ""
echo "Next steps:"
echo "  1. Upload training data to ~/chess-training/data/"
echo "  2. Run: ./train.sh --data ~/chess-training/data/training.jsonl"
