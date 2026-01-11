#!/usr/bin/env python3
"""
Chess LLM Fine-tuning Script (FIXED)

Key fix: Only compute loss on assistant response tokens, not the entire conversation.
Uses TRL's SFTTrainer which handles this automatically with chat templates.
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print("=" * 60)
print("Chess LLM Fine-tuning - FIXED VERSION")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    print("WARNING: No GPU available!")

# Configuration
DATA_PATH = Path.home() / "chess-training" / "data" / "training_elite_30k.jsonl"
OUTPUT_DIR = Path.home() / "chess-training" / "models-v2"
MODEL_NAME = "Qwen/Qwen2.5-3B"

EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5  # Slightly higher LR for faster convergence
LORA_RANK = 64
LORA_ALPHA = 128
MAX_SEQ_LENGTH = 512

print(f"\nConfiguration:")
print(f"  Data: {DATA_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Model: {MODEL_NAME}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  LoRA rank: {LORA_RANK}")
print(f"  Max seq length: {MAX_SEQ_LENGTH}")

# Create output dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
print("\n[1/5] Loading training data...")
with open(DATA_PATH, 'r') as f:
    examples = [json.loads(line) for line in f]
print(f"  Loaded {len(examples)} examples")

# Load tokenizer
print("\n[2/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Important for training

# Prepare dataset - keep messages format for SFTTrainer
dataset = Dataset.from_list(examples)

# Split into train/eval
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Load model with quantization
print("\n[3/5] Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # For compatibility
)
model = prepare_model_for_kbit_training(model)

# Configure LoRA
print("\n[4/5] Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# SFT Config - key settings for proper training
sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,  # Don't pack sequences
    dataset_text_field=None,  # We'll use messages format
)

# Create trainer
# SFTTrainer with chat template will automatically:
# 1. Apply chat template to messages
# 2. Mask prompt tokens (only compute loss on assistant response)
print("\n[5/5] Training...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# Train
trainer.train()

# Save final model
print("\nSaving final model...")
final_path = OUTPUT_DIR / "final"
trainer.save_model(str(final_path))
tokenizer.save_pretrained(str(final_path))

print("\n" + "=" * 60)
print("Training complete!")
print(f"Model saved to: {final_path}")
print("=" * 60)
