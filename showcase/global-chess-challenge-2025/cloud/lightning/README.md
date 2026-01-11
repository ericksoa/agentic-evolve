# Lightning.ai Training for Global Chess Challenge 2025

Training chess LLM using Lightning.ai Studios.

## Quick Start

### Option 1: Lightning Studio (Interactive)

1. Go to [lightning.ai](https://lightning.ai) and open a Studio
2. Select GPU: **L4** (budget) or **A10G** (faster) or **A100** (fastest)
3. Clone the repo and upload training data:

```bash
# In your Lightning Studio terminal
git clone <your-repo-url>
cd global-chess-challenge-2025

# Upload training data (from local machine via Lightning UI or CLI)
# Or copy from another location

# Run setup
cd cloud/lightning
chmod +x setup.sh
./setup.sh

# Start training
./train.sh --data ~/chess-training/data/training_full.jsonl
```

### Option 2: Lightning CLI (Programmatic)

```bash
# From your local machine
pip install lightning

# Login
lightning login

# Run training job
lightning run app train_app.py --cloud --name chess-training
```

## GPU Recommendations

| GPU | VRAM | Cost/hr | Use Case |
|-----|------|---------|----------|
| L4 | 24GB | ~$0.40 | Budget training, testing |
| A10G | 24GB | ~$0.60 | Good balance |
| A100-40G | 40GB | ~$1.50 | Fast training |
| A100-80G | 80GB | ~$2.00 | Large batches |

## Setup

The setup script installs:
- PyTorch with CUDA support
- Transformers, PEFT, bitsandbytes
- python-chess and Stockfish
- Flash Attention 2 (if supported)

```bash
./setup.sh
```

## Training

### Basic Training

```bash
./train.sh --data /path/to/training.jsonl
```

### Custom Parameters

```bash
./train.sh \
    --data /path/to/training.jsonl \
    --output ~/chess-models/run1 \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5 \
    --lora-rank 64
```

### Resume from Checkpoint

```bash
./train.sh \
    --data /path/to/training.jsonl \
    --resume ~/chess-models/run1/checkpoint-500
```

## Uploading Training Data

### Via Lightning UI
1. Open your Studio
2. Use the file browser to upload `training_full.jsonl`

### Via Lightning CLI
```bash
# From local machine
lightning cp data/training_full.jsonl /teamspace/studios/this_studio/chess-training/data/
```

### Via Direct SCP (if SSH enabled)
```bash
scp data/training_full.jsonl <studio-ssh>:~/chess-training/data/
```

## Downloading Results

### Via Lightning UI
1. Navigate to your model directory in the file browser
2. Right-click and download

### Via Lightning CLI
```bash
lightning cp /teamspace/studios/this_studio/chess-models/final ./local-model/
```

## Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f ~/chess-training/logs/*.log

# TensorBoard (Lightning has built-in support)
tensorboard --logdir ~/chess-models/runs
```

## Persistent Storage

Lightning Studios have persistent storage at:
- `~/` - Your home directory (persists)
- `/teamspace/` - Shared team storage

Save important files to these locations to persist across Studio restarts.

## Cost Optimization

1. **Use L4 for development** - Cheaper, sufficient for testing
2. **Stop Studio when idle** - Studios charge while running
3. **Use spot instances** - Check if available for your plan
4. **Prepare data locally** - Don't pay GPU time for data prep

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
./train.sh --data data.jsonl --batch-size 2

# Enable gradient checkpointing (already default)
```

### Slow Training
```bash
# Install Flash Attention
pip install flash-attn --no-build-isolation

# Use larger GPU
# Switch to A10G or A100 in Studio settings
```

### Studio Disconnects
- Your training will continue if you used `nohup` or `screen`
- Checkpoints are saved automatically
- Resume with `--resume` flag
