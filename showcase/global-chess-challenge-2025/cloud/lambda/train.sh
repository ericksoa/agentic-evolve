#!/bin/bash
# =============================================================================
# Global Chess Challenge 2025 - Training Script for Lambda Labs
# =============================================================================
#
# Wrapper script for training chess LLM with optimized settings for cloud GPUs.
#
# Usage:
#   ./train.sh --data /path/to/training.jsonl [options]
#
# Examples:
#   # Basic training
#   ./train.sh --data data/chess_training.jsonl
#
#   # Full customization
#   ./train.sh --data data/chess_training.jsonl \
#       --output models/chess_v2 \
#       --epochs 5 \
#       --batch-size 8 \
#       --lr 2e-5 \
#       --lora-rank 128
#
#   # Resume from checkpoint
#   ./train.sh --data data/chess_training.jsonl --resume checkpoints/checkpoint-1000
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
DATA_PATH=""
OUTPUT_DIR="${HOME}/chess-training/models/chess_llm_$(date +%Y%m%d_%H%M%S)"
BASE_MODEL="Qwen/Qwen2.5-3B"
EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE="1e-5"
LORA_RANK=64
LORA_ALPHA=128
MAX_SEQ_LENGTH=512
RESUME_FROM=""
USE_WANDB=false
WANDB_PROJECT="chess-llm"
NUM_GPUS=1
USE_DEEPSPEED=false

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient-accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            USE_WANDB=true
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --deepspeed)
            USE_DEEPSPEED=true
            shift
            ;;
        --help)
            echo "Usage: $0 --data <path> [options]"
            echo ""
            echo "Required:"
            echo "  --data PATH              Path to training data (JSONL format)"
            echo ""
            echo "Optional:"
            echo "  --output DIR             Output directory (default: timestamped)"
            echo "  --base-model MODEL       Base model (default: Qwen/Qwen2.5-3B)"
            echo "  --epochs N               Number of epochs (default: 3)"
            echo "  --batch-size N           Batch size per GPU (default: 4)"
            echo "  --gradient-accumulation N Gradient accumulation steps (default: 4)"
            echo "  --lr RATE                Learning rate (default: 1e-5)"
            echo "  --lora-rank N            LoRA rank (default: 64)"
            echo "  --lora-alpha N           LoRA alpha (default: 128)"
            echo "  --max-seq-length N       Max sequence length (default: 512)"
            echo "  --resume PATH            Resume from checkpoint"
            echo "  --wandb                  Enable Weights & Biases logging"
            echo "  --wandb-project NAME     W&B project name (default: chess-llm)"
            echo "  --num-gpus N             Number of GPUs (default: 1)"
            echo "  --deepspeed              Enable DeepSpeed for multi-GPU"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate Arguments
# -----------------------------------------------------------------------------
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data is required"
    echo "Usage: $0 --data <path> [options]"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${HOME}/chess-training"
LOG_FILE="${PROJECT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# -----------------------------------------------------------------------------
# Display Configuration
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Global Chess Challenge 2025 - Training"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data:                 ${DATA_PATH}"
echo "  Output:               ${OUTPUT_DIR}"
echo "  Base model:           ${BASE_MODEL}"
echo "  Epochs:               ${EPOCHS}"
echo "  Batch size:           ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "  Effective batch:      $((BATCH_SIZE * GRADIENT_ACCUMULATION * NUM_GPUS))"
echo "  Learning rate:        ${LEARNING_RATE}"
echo "  LoRA rank:            ${LORA_RANK}"
echo "  LoRA alpha:           ${LORA_ALPHA}"
echo "  Max sequence length:  ${MAX_SEQ_LENGTH}"
echo "  Number of GPUs:       ${NUM_GPUS}"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from:          ${RESUME_FROM}"
fi
if [ "$USE_WANDB" = true ]; then
    echo "  W&B project:          ${WANDB_PROJECT}"
fi
echo ""
echo "Log file: ${LOG_FILE}"
echo ""

# -----------------------------------------------------------------------------
# GPU Check
# -----------------------------------------------------------------------------
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Count examples
NUM_EXAMPLES=$(wc -l < "$DATA_PATH")
echo "Training data: ${NUM_EXAMPLES} examples"
echo ""

# -----------------------------------------------------------------------------
# Create Training Script
# -----------------------------------------------------------------------------
TRAIN_SCRIPT="${PROJECT_DIR}/run_training.py"

cat > "${TRAIN_SCRIPT}" << 'PYTHONEOF'
#!/usr/bin/env python3
"""
Cloud-optimized training script for Global Chess Challenge 2025.

This script includes optimizations for Lambda Labs GPU instances:
- Mixed precision training (fp16/bf16)
- Gradient checkpointing for memory efficiency
- Flash Attention 2 when available
- Proper LoRA configuration for chess task
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def load_training_data(data_path: str, tokenizer, max_length: int = 512):
    """Load and preprocess training data."""
    print(f"Loading training data from: {data_path}")

    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    def format_example(example):
        """Format a single example for training."""
        # Handle both message-based and raw text formats
        if "messages" in example:
            messages = example["messages"]
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            return {"text": text}
        elif "text" in example:
            return {"text": example["text"]}
        elif "fen" in example:
            # Raw training example from generate_data.py
            fen = example["fen"]
            best_move = example["best_move"]
            legal_moves = example.get("legal_moves", [])
            rationale = example.get("rationale", "Best move in this position.")

            user_content = f"""Position (FEN): {fen}
Side to move: {'White' if ' w ' in fen else 'Black'}
Legal moves: {', '.join(legal_moves[:20])}

Select the best move."""

            assistant_content = f"<uci_move>{best_move}</uci_move>\n<rationale>{rationale}</rationale>"

            text = f"""<|im_start|>system
You are a chess grandmaster. Given a position and legal moves, select the best move.<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
"""
            return {"text": text}
        else:
            raise ValueError(f"Unknown example format: {list(example.keys())}")

    # Convert to dataset
    formatted = [format_example(ex) for ex in examples]
    dataset = Dataset.from_list(formatted)

    # Tokenize
    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=["text"], num_proc=4)
    print(f"Tokenized {len(tokenized)} examples")

    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("CHESS LLM TRAINING - Cloud Edition")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine compute dtype based on GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        # Use bf16 for Ampere+ (A100, H100), fp16 for older (A10, V100)
        use_bf16 = any(x in gpu_name for x in ['a100', 'h100', 'a6000'])
        use_fp16 = not use_bf16
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        use_bf16 = False
        use_fp16 = True
        compute_dtype = torch.float16

    print(f"Using {'bf16' if use_bf16 else 'fp16'} precision")

    # Check for Flash Attention
    use_flash_attn = False
    try:
        import flash_attn
        use_flash_attn = True
        print("Flash Attention 2: Available")
    except ImportError:
        print("Flash Attention 2: Not available (training will be slower)")

    # Load tokenizer
    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load model
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }

    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    train_dataset = load_training_data(args.data, tokenizer, args.max_seq_length)

    # Split into train/eval
    split = train_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    print(f"\nTrain size: {len(train_data)}")
    print(f"Eval size: {len(eval_data)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=f"chess-{datetime.now().strftime('%Y%m%d-%H%M')}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    # Initialize W&B if requested
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                "base_model": args.base_model,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "max_seq_length": args.max_seq_length,
            }
        )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    final_path = Path(args.output) / "final"
    print(f"\nSaving final model to: {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save training config
    config_path = final_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "base_model": args.base_model,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_seq_length": args.max_seq_length,
            "data_path": args.data,
            "final_loss": trainer.state.best_metric,
            "total_steps": trainer.state.global_step,
            "training_time": str(datetime.now()),
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best eval loss: {trainer.state.best_metric:.4f}")
    print(f"Total steps: {trainer.state.global_step}")
    print(f"Model saved to: {final_path}")
    print(f"End time: {datetime.now().isoformat()}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
PYTHONEOF

chmod +x "${TRAIN_SCRIPT}"

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------
echo "Starting training..."
echo "=============================================="

# Build command
CMD="python3 ${TRAIN_SCRIPT}"
CMD="${CMD} --data ${DATA_PATH}"
CMD="${CMD} --output ${OUTPUT_DIR}"
CMD="${CMD} --base-model ${BASE_MODEL}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --gradient-accumulation ${GRADIENT_ACCUMULATION}"
CMD="${CMD} --lr ${LEARNING_RATE}"
CMD="${CMD} --lora-rank ${LORA_RANK}"
CMD="${CMD} --lora-alpha ${LORA_ALPHA}"
CMD="${CMD} --max-seq-length ${MAX_SEQ_LENGTH}"

if [ -n "$RESUME_FROM" ]; then
    CMD="${CMD} --resume ${RESUME_FROM}"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
fi

# Multi-GPU with accelerate
if [ "$NUM_GPUS" -gt 1 ]; then
    if [ "$USE_DEEPSPEED" = true ]; then
        CMD="accelerate launch --num_processes ${NUM_GPUS} --use_deepspeed ${TRAIN_SCRIPT}"
    else
        CMD="accelerate launch --num_processes ${NUM_GPUS} ${TRAIN_SCRIPT}"
    fi
    CMD="${CMD} --data ${DATA_PATH}"
    CMD="${CMD} --output ${OUTPUT_DIR}"
    CMD="${CMD} --base-model ${BASE_MODEL}"
    CMD="${CMD} --epochs ${EPOCHS}"
    CMD="${CMD} --batch-size ${BATCH_SIZE}"
    CMD="${CMD} --gradient-accumulation ${GRADIENT_ACCUMULATION}"
    CMD="${CMD} --lr ${LEARNING_RATE}"
    CMD="${CMD} --lora-rank ${LORA_RANK}"
    CMD="${CMD} --lora-alpha ${LORA_ALPHA}"
    CMD="${CMD} --max-seq-length ${MAX_SEQ_LENGTH}"
fi

echo "Command: ${CMD}"
echo ""

# Run with logging
${CMD} 2>&1 | tee "${LOG_FILE}"

# -----------------------------------------------------------------------------
# Post-Training
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Model saved to: ${OUTPUT_DIR}/final"
echo "Training log: ${LOG_FILE}"
echo ""
echo "To download the model:"
echo "  scp -r user@lambda-instance:${OUTPUT_DIR}/final ./chess_model/"
echo ""
echo "To evaluate the model, run:"
echo "  python3 evaluate_model.py --model ${OUTPUT_DIR}/final"
echo ""
