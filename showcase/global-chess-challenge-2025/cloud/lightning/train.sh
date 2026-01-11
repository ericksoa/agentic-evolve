#!/bin/bash
# Chess LLM Training Script for Lightning.ai

set -e

# Default values
DATA_PATH=""
OUTPUT_DIR="$HOME/chess-training/models/chess_llm_$(date +%Y%m%d_%H%M%S)"
EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE="2e-5"
LORA_RANK=64
LORA_ALPHA=128
MAX_SEQ_LENGTH=512
RESUME_FROM=""
WANDB_PROJECT=""
BASE_MODEL="Qwen/Qwen2.5-3B"

# Parse arguments
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
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required args
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data is required"
    echo "Usage: ./train.sh --data /path/to/training.jsonl [options]"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$HOME/chess-training/logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=============================================="
echo "Chess LLM Training - Lightning.ai"
echo "=============================================="
echo "Data:       $DATA_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Base Model: $BASE_MODEL"
echo "Epochs:     $EPOCHS"
echo "Batch Size: $BATCH_SIZE (x$GRADIENT_ACCUMULATION accumulation)"
echo "LR:         $LEARNING_RATE"
echo "LoRA:       rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "Log:        $LOG_FILE"
echo "=============================================="

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run training
python3 << 'TRAINING_SCRIPT'
import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path

# Training parameters from environment
DATA_PATH = os.environ.get("DATA_PATH")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B")
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", 4))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-5))
LORA_RANK = int(os.environ.get("LORA_RANK", 64))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 128))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", 512))
RESUME_FROM = os.environ.get("RESUME_FROM", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")

print(f"Loading libraries...")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Quantization config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print(f"Applying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load training data
print(f"Loading training data from {DATA_PATH}...")
with open(DATA_PATH, 'r') as f:
    raw_data = [json.loads(line) for line in f]

print(f"Loaded {len(raw_data)} examples")

# Format for training
def format_example(example):
    messages = example.get("messages", [])
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
    return text

# Tokenize
def tokenize(examples):
    texts = [format_example(ex) for ex in examples]
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

print(f"Tokenizing...")
formatted_data = [{"text": format_example(ex)} for ex in raw_data]
dataset = Dataset.from_list(formatted_data)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=4,
)

# Add labels
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="wandb" if WANDB_PROJECT else "tensorboard",
    run_name=f"chess_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Resume if specified
if RESUME_FROM:
    print(f"Resuming from {RESUME_FROM}...")
    trainer.train(resume_from_checkpoint=RESUME_FROM)
else:
    print(f"Starting training...")
    trainer.train()

# Save final model
print(f"Saving final model to {OUTPUT_DIR}/final...")
final_path = os.path.join(OUTPUT_DIR, "final")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

# Save training config
config = {
    "base_model": BASE_MODEL,
    "data_path": DATA_PATH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "max_seq_length": MAX_SEQ_LENGTH,
    "num_examples": len(raw_data),
    "timestamp": datetime.now().isoformat(),
}
with open(os.path.join(final_path, "training_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*50}")
print(f"Training complete!")
print(f"Model saved to: {final_path}")
print(f"{'='*50}")
TRAINING_SCRIPT

# Export environment variables for Python script
export DATA_PATH OUTPUT_DIR BASE_MODEL EPOCHS BATCH_SIZE GRADIENT_ACCUMULATION
export LEARNING_RATE LORA_RANK LORA_ALPHA MAX_SEQ_LENGTH RESUME_FROM WANDB_PROJECT

# Actually run the Python script
python3 -c "
import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path

DATA_PATH = '$DATA_PATH'
OUTPUT_DIR = '$OUTPUT_DIR'
BASE_MODEL = '$BASE_MODEL'
EPOCHS = $EPOCHS
BATCH_SIZE = $BATCH_SIZE
GRADIENT_ACCUMULATION = $GRADIENT_ACCUMULATION
LEARNING_RATE = $LEARNING_RATE
LORA_RANK = $LORA_RANK
LORA_ALPHA = $LORA_ALPHA
MAX_SEQ_LENGTH = $MAX_SEQ_LENGTH
RESUME_FROM = '$RESUME_FROM'
WANDB_PROJECT = '$WANDB_PROJECT'

print('Loading libraries...')
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Loading model with 4-bit quantization...')
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)

print(f'Applying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f'Loading training data from {DATA_PATH}...')
with open(DATA_PATH, 'r') as f:
    raw_data = [json.loads(line) for line in f]
print(f'Loaded {len(raw_data)} examples')

def format_example(example):
    messages = example.get('messages', [])
    text = ''
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            text += f'<|im_start|>system\n{content}<|im_end|>\n'
        elif role == 'user':
            text += f'<|im_start|>user\n{content}<|im_end|>\n'
        elif role == 'assistant':
            text += f'<|im_start|>assistant\n{content}<|im_end|>\n'
    return text

print('Tokenizing...')
formatted_data = [{'text': format_example(ex)} for ex in raw_data]
dataset = Dataset.from_list(formatted_data)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    num_proc=4,
)

def add_labels(examples):
    examples['labels'] = examples['input_ids'].copy()
    return examples

tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type='cosine',
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    optim='paged_adamw_8bit',
    report_to='tensorboard',
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

if RESUME_FROM:
    print(f'Resuming from {RESUME_FROM}...')
    trainer.train(resume_from_checkpoint=RESUME_FROM)
else:
    print('Starting training...')
    trainer.train()

print(f'Saving final model to {OUTPUT_DIR}/final...')
final_path = os.path.join(OUTPUT_DIR, 'final')
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

config = {
    'base_model': BASE_MODEL,
    'data_path': DATA_PATH,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'gradient_accumulation': GRADIENT_ACCUMULATION,
    'learning_rate': LEARNING_RATE,
    'lora_rank': LORA_RANK,
    'lora_alpha': LORA_ALPHA,
    'max_seq_length': MAX_SEQ_LENGTH,
    'num_examples': len(raw_data),
    'timestamp': datetime.now().isoformat(),
}
with open(os.path.join(final_path, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print()
print('=' * 50)
print('Training complete!')
print(f'Model saved to: {final_path}')
print('=' * 50)
" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training finished. Log saved to: $LOG_FILE"
