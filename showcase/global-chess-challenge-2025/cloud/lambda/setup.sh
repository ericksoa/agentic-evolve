#!/bin/bash
# =============================================================================
# Global Chess Challenge 2025 - Lambda Labs Environment Setup
# =============================================================================
#
# This script sets up a Lambda Labs GPU instance for training the chess LLM.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Recommended instances:
#   - 1x A10 (24GB) - Budget option, good for testing
#   - 1x A100 (40GB/80GB) - Best for training
#   - 1x H100 (80GB) - Fastest training
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Global Chess Challenge 2025 - Setup Script"
echo "=============================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL="Qwen/Qwen2.5-3B"
PROJECT_DIR="${HOME}/chess-training"
DATA_DIR="${PROJECT_DIR}/data"
MODELS_DIR="${PROJECT_DIR}/models"
LOGS_DIR="${PROJECT_DIR}/logs"

# -----------------------------------------------------------------------------
# System Information
# -----------------------------------------------------------------------------
echo ""
echo "[1/7] System Information"
echo "------------------------"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "Warning: nvidia-smi not found"
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'Not found')"

# -----------------------------------------------------------------------------
# Create Directory Structure
# -----------------------------------------------------------------------------
echo ""
echo "[2/7] Creating Directory Structure"
echo "-----------------------------------"
mkdir -p "${DATA_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${LOGS_DIR}"
mkdir -p "${PROJECT_DIR}/configs"
mkdir -p "${PROJECT_DIR}/checkpoints"
echo "Created: ${PROJECT_DIR}"
echo "  - data/"
echo "  - models/"
echo "  - logs/"
echo "  - configs/"
echo "  - checkpoints/"

# -----------------------------------------------------------------------------
# Install System Dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] Installing System Dependencies"
echo "-------------------------------------"

# Stockfish for evaluation
if ! command -v stockfish &> /dev/null; then
    echo "Installing Stockfish..."
    sudo apt-get update -qq
    sudo apt-get install -y stockfish
else
    echo "Stockfish already installed"
fi

# -----------------------------------------------------------------------------
# Python Environment
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Setting Up Python Environment"
echo "-------------------------------------"

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support (if not already present)
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
}

# Install requirements
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r "${SCRIPT_DIR}/requirements.txt"
else
    echo "Installing Python dependencies (inline)..."
    pip install transformers>=4.36.0 accelerate>=0.25.0 peft>=0.7.0 \
                bitsandbytes>=0.41.0 datasets>=2.16.0 python-chess>=1.10.0 \
                trl>=0.7.0 wandb>=0.16.0 tensorboard>=2.15.0 \
                pandas numpy tqdm stockfish python-dotenv huggingface_hub
fi

# -----------------------------------------------------------------------------
# Install Flash Attention (optional but recommended)
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Installing Flash Attention 2"
echo "------------------------------------"
python3 -c "import flash_attn" 2>/dev/null && echo "Flash Attention already installed" || {
    echo "Installing Flash Attention 2 (this may take a few minutes)..."
    pip install flash-attn --no-build-isolation 2>/dev/null || {
        echo "Warning: Flash Attention installation failed. Training will work but may be slower."
    }
}

# -----------------------------------------------------------------------------
# Download Base Model
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] Downloading Base Model"
echo "-----------------------------"
echo "Model: ${BASE_MODEL}"

python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "${BASE_MODEL}"
cache_dir = "${MODELS_DIR}/base"

print(f"Downloading {model_name}...")

# Download tokenizer
print("  Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir
)

# Download model (just to cache it)
print("  Downloading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir
)

print(f"Model downloaded to: {cache_dir}")
print(f"Model parameters: {model.num_parameters():,}")

# Cleanup to free memory
del model
import gc
gc.collect()
import torch
torch.cuda.empty_cache()
EOF

# -----------------------------------------------------------------------------
# Verify Installation
# -----------------------------------------------------------------------------
echo ""
echo "[7/7] Verifying Installation"
echo "-----------------------------"

python3 << 'EOF'
import sys

def check_import(name, package=None):
    package = package or name
    try:
        __import__(package)
        print(f"  [OK] {name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("Checking required packages:")
checks = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
    ("datasets", "datasets"),
    ("python-chess", "chess"),
    ("trl", "trl"),
    ("huggingface_hub", "huggingface_hub"),
]

all_ok = all(check_import(name, pkg) for name, pkg in checks)

# Check CUDA
import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check Stockfish
import shutil
stockfish_path = shutil.which("stockfish")
print(f"\nStockfish: {'Found at ' + stockfish_path if stockfish_path else 'Not found'}")

if not all_ok:
    print("\nWarning: Some packages failed to install")
    sys.exit(1)
else:
    print("\nAll packages installed successfully!")
EOF

# -----------------------------------------------------------------------------
# Create Helper Scripts
# -----------------------------------------------------------------------------
echo ""
echo "Creating helper scripts..."

# Create a quick test script
cat > "${PROJECT_DIR}/test_setup.py" << 'TESTEOF'
#!/usr/bin/env python3
"""Quick test to verify the setup is working."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess

print("Testing chess LLM setup...")

# Test chess
board = chess.Board()
print(f"Chess library: OK (starting FEN: {board.fen()[:30]}...)")

# Test model loading
print("Testing model loading (this will use GPU memory)...")
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Test inference
prompt = "What is the best first move in chess?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Model inference: OK")
print(f"Sample response: {response[:100]}...")

print("\nSetup verification complete!")
TESTEOF
chmod +x "${PROJECT_DIR}/test_setup.py"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Project directory: ${PROJECT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Upload your training data (PGN or JSONL) to ${DATA_DIR}/"
echo "  2. Run training with: ./train.sh --data ${DATA_DIR}/your_data.jsonl"
echo "  3. Or run the full pipeline: ./run_full_pipeline.sh"
echo ""
echo "Optional: Test the setup with:"
echo "  python3 ${PROJECT_DIR}/test_setup.py"
echo ""
echo "Useful commands:"
echo "  nvidia-smi                    # Monitor GPU usage"
echo "  htop                          # Monitor CPU/RAM"
echo "  tail -f ${LOGS_DIR}/*.log     # Watch training logs"
echo ""
