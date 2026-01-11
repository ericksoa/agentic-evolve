# Cloud Training for Global Chess Challenge 2025

This directory contains scripts for training the chess LLM on Lambda Labs GPU instances.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Uploading Training Data](#uploading-training-data)
- [Starting a GPU Instance](#starting-a-gpu-instance)
- [Running Training](#running-training)
- [Downloading the Model](#downloading-the-model)
- [Cost Estimates](#cost-estimates)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. A Lambda Labs account with GPU access
2. SSH key configured for Lambda Labs
3. Training data (PGN file or pre-processed JSONL)

## Quick Start

```bash
# 1. Launch a Lambda Labs instance (A10, A100, or H100)

# 2. SSH into your instance
ssh ubuntu@<instance-ip>

# 3. Clone/upload the project
git clone <your-repo-url>
cd global-chess-challenge-2025/cloud

# 4. Run setup (one-time)
chmod +x setup.sh
./setup.sh

# 5. Upload your training data
# (from your local machine)
scp games.pgn ubuntu@<instance-ip>:~/chess-training/data/

# 6. Run the full pipeline
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh --pgn ~/chess-training/data/games.pgn
```

## Uploading Training Data

### Option 1: Upload PGN File

From your local machine:

```bash
# Single PGN file
scp path/to/games.pgn ubuntu@<instance-ip>:~/chess-training/data/

# Multiple files
scp -r path/to/pgn_folder/ ubuntu@<instance-ip>:~/chess-training/data/
```

### Option 2: Upload Pre-processed JSONL

If you already have training data in JSONL format:

```bash
scp training_data.jsonl ubuntu@<instance-ip>:~/chess-training/data/
```

### Option 3: Download from Cloud Storage

```bash
# From Google Drive (using gdown)
pip install gdown
gdown <file-id> -O ~/chess-training/data/games.pgn

# From AWS S3
aws s3 cp s3://bucket/games.pgn ~/chess-training/data/

# From Hugging Face
huggingface-cli download <dataset-name> --local-dir ~/chess-training/data/
```

### Data Format

**PGN Format** (recommended for raw games):
```
[Event "Tournament"]
[White "Player1"]
[Black "Player2"]
[WhiteElo "2400"]
[BlackElo "2350"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 ...
```

**JSONL Format** (for pre-processed data):
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Starting a GPU Instance

### Recommended Instances

| Instance Type | GPU Memory | Use Case | Cost/Hour |
|--------------|------------|----------|-----------|
| 1x A10 | 24 GB | Testing, small datasets | ~$0.60 |
| 1x A100 40GB | 40 GB | Production training | ~$1.29 |
| 1x A100 80GB | 80 GB | Large batch sizes | ~$1.99 |
| 1x H100 | 80 GB | Fastest training | ~$2.49 |

### Launch Steps

1. Go to [Lambda Labs Cloud](https://cloud.lambdalabs.com/)
2. Click "Launch Instance"
3. Select your preferred GPU type
4. Choose Ubuntu 22.04 with CUDA
5. Select your SSH key
6. Click "Launch"

### Connect to Instance

```bash
# Get the IP from Lambda Labs dashboard
ssh ubuntu@<instance-ip>
```

## Running Training

### Setup (First Time Only)

```bash
cd global-chess-challenge-2025/cloud
chmod +x *.sh
./setup.sh
```

This will:
- Install all Python dependencies
- Install Stockfish for evaluation
- Download the base model (Qwen/Qwen2.5-3B)
- Create the directory structure

### Option 1: Full Pipeline (Recommended)

Extracts data from PGN, trains, and evaluates:

```bash
./run_full_pipeline.sh --pgn ~/chess-training/data/games.pgn
```

With custom settings:

```bash
./run_full_pipeline.sh \
    --pgn ~/chess-training/data/games.pgn \
    --max-examples 100000 \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --lora-rank 128
```

### Option 2: Training Only

If you already have JSONL training data:

```bash
./train.sh --data ~/chess-training/data/training.jsonl
```

With custom settings:

```bash
./train.sh \
    --data ~/chess-training/data/training.jsonl \
    --output ~/chess-training/models/my_model \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-5 \
    --lora-rank 64 \
    --lora-alpha 128
```

### Monitoring Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f ~/chess-training/logs/*.log

# View TensorBoard (if enabled)
tensorboard --logdir ~/chess-training/models/*/runs
```

### Resume from Checkpoint

If training is interrupted:

```bash
./train.sh \
    --data ~/chess-training/data/training.jsonl \
    --resume ~/chess-training/models/chess_llm_*/checkpoint-500
```

## Downloading the Model

### Download Final Model

From your local machine:

```bash
# Download final model
scp -r ubuntu@<instance-ip>:~/chess-training/models/chess_llm_*/final ./chess_model/

# Download with all checkpoints
scp -r ubuntu@<instance-ip>:~/chess-training/models/chess_llm_*/ ./chess_model_full/
```

### Download Results

```bash
scp ubuntu@<instance-ip>:~/chess-training/results/*.json ./results/
scp ubuntu@<instance-ip>:~/chess-training/logs/*.log ./logs/
```

### Using the Model Locally

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./chess_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./chess_model")

# Generate move
prompt = """<|im_start|>system
You are a chess grandmaster. Given a position and legal moves, select the best move.<|im_end|>
<|im_start|>user
Position (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Side to move: Black
Legal moves: a7a6, a7a5, b7b6, b7b5, c7c6, c7c5, d7d6, d7d5, e7e6, e7e5, f7f6, f7f5, g7g6, g7g5, h7h6, h7h5, b8a6, b8c6, g8f6, g8h6

Select the best move.<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Cost Estimates

### Training Costs

Based on Lambda Labs pricing (as of 2025):

| Dataset Size | GPU | Training Time | Est. Cost |
|--------------|-----|---------------|-----------|
| 10K examples | A10 | ~30 min | ~$0.30 |
| 50K examples | A10 | ~2 hours | ~$1.20 |
| 50K examples | A100 | ~1 hour | ~$1.30 |
| 100K examples | A100 | ~2 hours | ~$2.60 |
| 100K examples | H100 | ~1 hour | ~$2.50 |

### Tips to Reduce Costs

1. **Test locally first**: Use a small subset to verify your pipeline
2. **Use spot instances**: Up to 50% cheaper when available
3. **Prepare data in advance**: Don't pay for GPU time during data prep
4. **Use 4-bit quantization**: Already enabled in our scripts
5. **Monitor training**: Stop early if loss plateaus

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
./train.sh --data data.jsonl --batch-size 2

# Increase gradient accumulation to maintain effective batch size
./train.sh --data data.jsonl --batch-size 2 --gradient-accumulation 8
```

### Slow Training

1. Install Flash Attention 2:
```bash
pip install flash-attn --no-build-isolation
```

2. Increase batch size if GPU memory allows
3. Use a larger GPU (A100 vs A10)

### CUDA Out of Memory During Evaluation

The evaluation script loads the model fully. If you run out of memory:

```bash
# Skip evaluation and run it separately on a fresh instance
./run_full_pipeline.sh --pgn games.pgn --skip-eval
```

### Stockfish Not Found

```bash
# Install Stockfish
sudo apt-get update && sudo apt-get install -y stockfish

# Or download manually
wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64.tar
tar -xf stockfish-ubuntu-x86-64.tar
sudo mv stockfish/stockfish-ubuntu-x86-64 /usr/local/bin/stockfish
```

### Model Not Loading

```bash
# Clear Hugging Face cache and re-download
rm -rf ~/.cache/huggingface/
./setup.sh
```

### Training Interrupted

Your checkpoints are saved automatically. Resume with:

```bash
./train.sh --data data.jsonl --resume /path/to/checkpoint-XXX
```

## Advanced Configuration

### Multi-GPU Training

For instances with multiple GPUs:

```bash
./train.sh --data data.jsonl --num-gpus 4 --batch-size 4
# Effective batch size = 4 GPUs * 4 batch * 4 accumulation = 64
```

### DeepSpeed (Experimental)

For very large models or datasets:

```bash
./train.sh --data data.jsonl --deepspeed --num-gpus 4
```

### Custom LoRA Configuration

```bash
./train.sh \
    --data data.jsonl \
    --lora-rank 128 \    # Higher rank = more capacity, more VRAM
    --lora-alpha 256     # Usually 2x rank
```

### Weights & Biases Integration

```bash
# Login first
wandb login

# Enable logging
./train.sh --data data.jsonl --wandb --wandb-project chess-llm-experiment
```

## File Structure

After running the pipeline:

```
~/chess-training/
├── data/
│   └── chess_llm_YYYYMMDD_training.jsonl
├── models/
│   └── chess_llm_YYYYMMDD_HHMMSS/
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── final/
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           ├── tokenizer.json
│           └── training_config.json
├── results/
│   └── chess_llm_YYYYMMDD_results.json
└── logs/
    └── chess_llm_YYYYMMDD.log
```

## Support

- Lambda Labs Docs: https://docs.lambdalabs.com/
- Hugging Face PEFT: https://huggingface.co/docs/peft/
- Transformers: https://huggingface.co/docs/transformers/
- Project Issues: [GitHub Issues]
