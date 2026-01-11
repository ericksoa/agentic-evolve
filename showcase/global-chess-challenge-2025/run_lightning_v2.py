#!/usr/bin/env python3
"""
Run chess LLM training on Lightning.ai - FIXED VERSION

Uses SFTTrainer which properly masks prompt tokens.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Lightning.ai credentials
LIGHTNING_USER_ID = os.getenv("LIGHTNING_USER_ID")
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY")
LIGHTNING_USERNAME = os.getenv("LIGHTNING_AI_USERNAME")
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")

# Set environment variables for Lightning SDK
os.environ["LIGHTNING_USER_ID"] = LIGHTNING_USER_ID
os.environ["LIGHTNING_API_KEY"] = LIGHTNING_API_KEY

from lightning_sdk import Studio, Machine

# Training script to run on Lightning.ai
TRAIN_SCRIPT = '''#!/usr/bin/env python3
"""Chess LLM Fine-tuning - FIXED with SFTTrainer"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print("=" * 60)
print("Chess LLM Fine-tuning - FIXED VERSION (SFTTrainer)")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    print("WARNING: No GPU available!")
    sys.exit(1)

# Configuration
DATA_PATH = Path.home() / "chess-training" / "data" / "training_elite_30k.jsonl"
OUTPUT_DIR = Path.home() / "chess-training" / "models-v2"
MODEL_NAME = "Qwen/Qwen2.5-3B"

EPOCHS = 3
BATCH_SIZE = 2  # Reduced from 4 to avoid OOM
LEARNING_RATE = 2e-5
LORA_RANK = 64
LORA_ALPHA = 128
MAX_SEQ_LENGTH = 512

print(f"\\nConfiguration:")
print(f"  Data: {DATA_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Model: {MODEL_NAME}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  LoRA rank: {LORA_RANK}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
print("\\n[1/5] Loading training data...")
with open(DATA_PATH, 'r') as f:
    examples = [json.loads(line) for line in f]
print(f"  Loaded {len(examples)} examples")

# Load tokenizer
print("\\n[2/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare dataset
dataset = Dataset.from_list(examples)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Load model
print("\\n[3/5] Loading model...")
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
    attn_implementation="eager",
)
model = prepare_model_for_kbit_training(model)

# Configure LoRA
print("\\n[4/5] Configuring LoRA...")
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

# SFT Config - KEY: completion_only_loss=True to only train on assistant response
sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,  # Doubled since batch_size halved
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
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    completion_only_loss=True,
)

# Train with SFTTrainer (handles prompt masking automatically)
print("\\n[5/5] Training with SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()

# Save
print("\\nSaving final model...")
final_path = OUTPUT_DIR / "final"
trainer.save_model(str(final_path))
tokenizer.save_pretrained(str(final_path))

print("\\n" + "=" * 60)
print("Training complete!")
print(f"Model saved to: {final_path}")
print("=" * 60)
'''


def run_training():
    """Run the fixed training on Lightning.ai"""
    print()
    print("=" * 60)
    print("Chess LLM Training V2 - Lightning.ai Runner (FIXED)")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  Data: data/training_elite_30k.jsonl")
    print("  GPU: L4")
    print("  Fix: Using SFTTrainer with proper prompt masking")
    print()

    # Create/connect to studio
    studio_name = "chess-llm-v4"
    print(f"[1/6] Creating Lightning Studio: {studio_name}")
    print(f"       Teamspace: {LIGHTNING_TEAMSPACE}")
    print()

    studio = Studio(
        name=studio_name,
        teamspace=LIGHTNING_TEAMSPACE,
        user=LIGHTNING_USERNAME,
        create_ok=True
    )

    # Start GPU
    print("[2/6] Starting L4 GPU instance...")
    studio.start(Machine.L4)
    print("  L4 GPU started!")
    print()

    # Install dependencies (including TRL)
    print("[3/6] Installing dependencies...")
    commands = [
        "pip install --upgrade pip -q",
        "pip install torch --index-url https://download.pytorch.org/whl/cu121 -q",
        "pip install transformers datasets accelerate peft bitsandbytes -q",
        "pip install trl -q",  # TRL for SFTTrainer
        "pip install python-chess -q",
    ]
    for cmd in commands:
        print(f"  Running: {cmd[:60]}...")
        studio.run(cmd)
    print()

    # Upload training data (if not already there)
    print("[4/6] Checking training data...")
    studio.run("mkdir -p ~/chess-training/data")

    # Check if data exists
    check = studio.run("ls ~/chess-training/data/training_elite_30k.jsonl 2>/dev/null || echo 'NOT_FOUND'")
    if "NOT_FOUND" in check:
        print("  Uploading training data...")
        data_path = Path(__file__).parent / "data" / "training_elite_30k.jsonl"
        studio.upload_file(str(data_path), "chess-training/data/training_elite_30k.jsonl")
        print("  Uploaded!")
    else:
        print("  Training data already exists")
    print()

    # Upload and run training script
    print("[5/6] Starting training...")

    # Write training script
    studio.run(f"cat > ~/train_chess_v2.py << 'SCRIPT_EOF'\n{TRAIN_SCRIPT}\nSCRIPT_EOF")

    # Run training in background with nohup
    studio.run("nohup python ~/train_chess_v2.py > ~/train_chess_v2.log 2>&1 &")
    print("  Training started in background!")
    print("  Log file: ~/train_chess_v2.log")
    print()

    # Wait a moment and check if it started
    import time
    time.sleep(10)

    # Check if training is running
    ps_output = studio.run("ps aux | grep train_chess_v2 | grep -v grep || echo 'NOT_RUNNING'")
    if "NOT_RUNNING" in ps_output:
        print("WARNING: Training process not found!")
        # Check log for errors
        log_output = studio.run("cat ~/train_chess_v2.log 2>/dev/null || echo 'No log yet'")
        print(f"Log output:\n{log_output}")
    else:
        print("  Training process is running!")
        # Show initial log output
        log_output = studio.run("head -50 ~/train_chess_v2.log 2>/dev/null || echo 'No log yet'")
        print(f"\nInitial log:\n{log_output}")

    print()
    print("[6/6] Training launched!")
    print()
    print("To check progress:")
    print("  studio.run('tail -100 ~/train_chess_v2.log')")
    print("  studio.run('ps aux | grep train_chess')")

    return studio


if __name__ == "__main__":
    studio = run_training()
    print()
    print("Studio is still running. To stop:")
    print(f"  studio.stop()")
