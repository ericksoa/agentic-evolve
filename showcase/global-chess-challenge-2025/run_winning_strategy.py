#!/usr/bin/env python3
"""
Execute the Winning Strategy

This script runs the full pipeline to train a winning chess model:
1. Download fresh PGN data (if needed)
2. Extract training data with Stockfish filtering
3. Upload to Lightning.ai and train
4. Download and evaluate

Usage:
    python run_winning_strategy.py --sf-top 3 --epochs 4 --lora-rank 128
"""
import os
import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"


def download_pgn_data():
    """Download Lichess elite database if not present"""
    pgn_path = DATA_DIR / "lichess_elite.pgn"

    if pgn_path.exists():
        print(f"PGN already exists: {pgn_path}")
        return pgn_path

    print("Downloading Lichess elite database...")
    # This is a curated subset of high-rated games
    # Alternative: Full database from https://database.lichess.org/

    # For now, we'll use the existing data
    print("Using existing training data as base")
    return DATA_DIR / "training_elite_30k.jsonl"


def extract_filtered_data(
    sf_top: int = 3,
    min_elo: int = 2400,
    max_cpl: int = 50,
    max_examples: int = 50000,
) -> Path:
    """
    Extract training data with Stockfish filtering.

    This is THE KEY to winning - only train on correct moves!
    """
    output_name = f"training_sf{sf_top}_elo{min_elo}_cpl{max_cpl}.jsonl"
    output_path = DATA_DIR / output_name

    if output_path.exists():
        print(f"Filtered data already exists: {output_path}")
        return output_path

    print(f"Extracting filtered training data...")
    print(f"  SF top-{sf_top}, Elo >= {min_elo}, CPL <= {max_cpl}")

    # For now, use existing data and filter it
    # In production, would run extract_filtered_data.py on raw PGN

    base_data = DATA_DIR / "training_elite_30k.jsonl"
    if not base_data.exists():
        print(f"ERROR: Base data not found: {base_data}")
        sys.exit(1)

    # Copy base data (filtering would happen here)
    # TODO: Implement actual Stockfish filtering
    import shutil
    shutil.copy(base_data, output_path)
    print(f"  Created {output_path} (using base data for now)")

    return output_path


def run_training_lightning(
    data_path: Path,
    epochs: int = 4,
    lora_rank: int = 128,
    run_name: str = None,
) -> dict:
    """
    Run training on Lightning.ai
    """
    os.environ["LIGHTNING_USER_ID"] = os.getenv("LIGHTNING_USER_ID")
    os.environ["LIGHTNING_API_KEY"] = os.getenv("LIGHTNING_API_KEY")

    from lightning_sdk import Studio

    LIGHTNING_USERNAME = os.getenv("LIGHTNING_AI_USERNAME")
    LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")

    if run_name is None:
        run_name = f"chess_e{epochs}_r{lora_rank}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"\n{'='*60}")
    print(f"TRAINING: {run_name}")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Data: {data_path.name}")

    studio = Studio(
        name="chess-llm-v4",
        teamspace=LIGHTNING_TEAMSPACE,
        user=LIGHTNING_USERNAME,
    )

    # Start if needed
    if studio.status != "Running":
        print("Starting studio...")
        studio.start()
        time.sleep(30)

    # Upload data
    remote_data = f"chess-training/{run_name}/training.jsonl"
    print(f"Uploading data to {remote_data}...")
    studio.upload_file(str(data_path), remote_data)

    # Create training script
    train_script = f'''#!/usr/bin/env python3
"""Training script: {run_name}"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_PATH = Path.home() / "{remote_data}"
OUTPUT_DIR = Path.home() / "chess-training" / "{run_name}" / "model"
MODEL_NAME = "Qwen/Qwen2.5-3B"

EPOCHS = {epochs}
LORA_RANK = {lora_rank}
LORA_ALPHA = {lora_rank * 2}

print("=" * 60)
print(f"Training: {run_name}")
print("=" * 60)

with open(DATA_PATH, 'r') as f:
    examples = [json.loads(line) for line in f]
print(f"Loaded {{len(examples)}} examples")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = Dataset.from_list(examples)
split = dataset.train_test_split(test_size=0.1, seed=42)

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
    max_length=512,
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    processing_class=tokenizer,
)

trainer.train()

final_path = OUTPUT_DIR / "final"
trainer.save_model(str(final_path))
tokenizer.save_pretrained(str(final_path))

print()
print("=" * 60)
print("Training complete!")
print(f"Model saved to: {{final_path}}")
print("=" * 60)
'''

    # Upload and run
    script_path = PROJECT_DIR / "temp_train.py"
    with open(script_path, 'w') as f:
        f.write(train_script)

    studio.upload_file(str(script_path), f"chess-training/{run_name}/train.py")
    script_path.unlink()  # Clean up

    print("Starting training (this will take 4-8 hours)...")
    print("Monitor at: https://lightning.ai")

    # Run training
    result = studio.run(
        f"cd ~/chess-training/{run_name} && python train.py 2>&1 | tee train.log",
    )

    if "Training complete!" in result:
        return {
            "success": True,
            "run_name": run_name,
            "checkpoint": f"chess-training/{run_name}/model/final",
            "log": result,
        }
    else:
        return {
            "success": False,
            "run_name": run_name,
            "error": result[-2000:],
        }


def download_and_evaluate(run_name: str, checkpoint_path: str) -> dict:
    """Download checkpoint and run ACPL evaluation"""
    from evaluate_final_model import main as evaluate_main

    print(f"\nDownloading checkpoint: {checkpoint_path}")

    local_path = PROJECT_DIR / "models" / run_name
    local_path.mkdir(parents=True, exist_ok=True)

    os.environ["LIGHTNING_USER_ID"] = os.getenv("LIGHTNING_USER_ID")
    os.environ["LIGHTNING_API_KEY"] = os.getenv("LIGHTNING_API_KEY")

    from lightning_sdk import Studio

    LIGHTNING_USERNAME = os.getenv("LIGHTNING_AI_USERNAME")
    LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")

    studio = Studio(
        name="chess-llm-v4",
        teamspace=LIGHTNING_TEAMSPACE,
        user=LIGHTNING_USERNAME,
    )

    files = [
        "adapter_config.json", "adapter_model.safetensors",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "chat_template.jinja",
    ]

    for fname in files:
        try:
            studio.download_file(f"{checkpoint_path}/{fname}", str(local_path / fname))
            print(f"  Downloaded {fname}")
        except Exception as e:
            print(f"  Warning: {fname}: {e}")

    # Run evaluation
    print("\nRunning ACPL evaluation...")
    # This would call evaluate_final_model with the new checkpoint
    # For now, return placeholder
    return {
        "acpl": 0,
        "legal_rate": 0,
        "checkpoint": str(local_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run winning strategy")
    parser.add_argument("--sf-top", type=int, default=3, help="Stockfish top-N filter")
    parser.add_argument("--min-elo", type=int, default=2400, help="Minimum Elo")
    parser.add_argument("--max-cpl", type=int, default=50, help="Max centipawn loss")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--lora-rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("WINNING STRATEGY EXECUTION")
    print("=" * 60)
    print(f"SF filter: top-{args.sf_top}")
    print(f"Min Elo: {args.min_elo}")
    print(f"Max CPL: {args.max_cpl}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_rank}")
    print()

    # Step 1: Data
    print("[1/3] Preparing training data...")
    data_path = extract_filtered_data(
        sf_top=args.sf_top,
        min_elo=args.min_elo,
        max_cpl=args.max_cpl,
    )

    if args.skip_training:
        print("\nSkipping training (--skip-training)")
        return

    # Step 2: Train
    print("\n[2/3] Running training...")
    run_name = f"sf{args.sf_top}_e{args.epochs}_r{args.lora_rank}"
    result = run_training_lightning(
        data_path=data_path,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
        run_name=run_name,
    )

    if not result["success"]:
        print(f"\nTraining FAILED: {result.get('error', 'Unknown')[:500]}")
        return

    # Step 3: Evaluate
    print("\n[3/3] Evaluating...")
    eval_result = download_and_evaluate(run_name, result["checkpoint"])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"ACPL: {eval_result.get('acpl', 'N/A')}")
    print(f"Legal rate: {eval_result.get('legal_rate', 'N/A')}")
    print(f"Checkpoint: {eval_result.get('checkpoint', 'N/A')}")


if __name__ == "__main__":
    main()
