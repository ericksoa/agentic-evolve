#!/usr/bin/env python3
"""
Run Chess LLM Training on Lightning.ai GPU.

Uses Lightning.ai Studios for GPU-accelerated fine-tuning of Qwen2.5-3B.

Setup:
    1. Create account at https://lightning.ai/sign-up
    2. Get credentials from https://lightning.ai/<username>/home?settings=keys
    3. Create .env file with:
       LIGHTNING_USER_ID=your-user-id
       LIGHTNING_API_KEY=your-api-key
       LIGHTNING_AI_USERNAME=your-username
       LIGHTNING_TEAMSPACE=your-teamspace

Usage:
    pip install lightning-sdk python-dotenv
    python run_lightning.py --data data/training_elite.jsonl --epochs 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from textwrap import dedent

# Load .env file if present
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

try:
    from lightning_sdk import Studio, Machine
except ImportError:
    print("Error: lightning-sdk not installed")
    print("Run: pip install lightning-sdk")
    sys.exit(1)


# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def check_credentials():
    """Check that Lightning.ai credentials are set."""
    required = ["LIGHTNING_USER_ID", "LIGHTNING_API_KEY", "LIGHTNING_AI_USERNAME", "LIGHTNING_TEAMSPACE"]
    missing = [var for var in required if not os.environ.get(var)]

    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nGet your credentials from:")
        print("  https://lightning.ai/<username>/home?settings=keys")
        return False
    return True


def get_training_script(
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 64,
) -> str:
    """Generate the training script to run on Lightning."""
    return dedent(f'''
        #!/usr/bin/env python3
        """Chess LLM Fine-tuning Script"""
        import json
        import torch
        from pathlib import Path
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("=" * 60)
        print("Chess LLM Fine-tuning - Lightning.ai")
        print("=" * 60)

        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {{gpu_name}} ({{gpu_memory:.1f}} GB)")
        else:
            print("WARNING: No GPU available!")

        # Configuration
        DATA_PATH = "{data_path}"
        OUTPUT_DIR = "{output_dir}"
        MODEL_NAME = "Qwen/Qwen2.5-3B"

        EPOCHS = {epochs}
        BATCH_SIZE = {batch_size}
        LEARNING_RATE = {learning_rate}
        LORA_RANK = {lora_rank}
        LORA_ALPHA = {lora_rank * 2}

        print(f"\\nConfiguration:")
        print(f"  Data: {{DATA_PATH}}")
        print(f"  Output: {{OUTPUT_DIR}}")
        print(f"  Epochs: {{EPOCHS}}")
        print(f"  Batch size: {{BATCH_SIZE}}")
        print(f"  Learning rate: {{LEARNING_RATE}}")
        print(f"  LoRA rank: {{LORA_RANK}}")

        # Load training data
        print("\\n[1/5] Loading training data...")
        with open(DATA_PATH, 'r') as f:
            examples = [json.loads(line) for line in f]
        print(f"  Loaded {{len(examples)}} examples")

        # Load tokenizer
        print("\\n[2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare dataset
        def format_example(example):
            messages = example.get("messages", [])
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {{"text": text}}

        dataset = Dataset.from_list(examples)
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_dataset = tokenized_dataset.map(
            lambda x: {{"labels": x["input_ids"].copy()}},
            batched=False
        )

        # Split into train/eval
        split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {{len(train_dataset)}}, Eval: {{len(eval_dataset)}}")

        # Load model with quantization
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

        # Training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
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
            optim="adamw_torch",
            report_to="none",
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )

        # Train
        print("\\n[5/5] Training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        # Save final model
        print("\\nSaving model...")
        trainer.save_model(OUTPUT_DIR + "/final")
        tokenizer.save_pretrained(OUTPUT_DIR + "/final")

        print("\\n" + "=" * 60)
        print("Training complete!")
        print(f"Model saved to: {{OUTPUT_DIR}}/final")
        print("=" * 60)
    ''')


def run_training_on_lightning(
    data_file: str,
    studio_name: str = "chess-llm-training",
    teamspace: str = None,
    machine: str = "L4",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 64,
):
    """
    Run chess LLM training on Lightning.ai GPU.

    Args:
        data_file: Path to training data JSONL file
        studio_name: Name for the Lightning studio
        teamspace: Lightning teamspace to use
        machine: GPU type (T4, L4, A10G, A100)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        lora_rank: LoRA rank
    """
    if not check_credentials():
        return None

    data_path = Path(data_file)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_file}")
        return None

    username = os.environ["LIGHTNING_AI_USERNAME"]
    teamspace = teamspace or os.environ.get("LIGHTNING_TEAMSPACE")

    # Select machine type
    machine_map = {
        "T4": Machine.T4,
        "L4": Machine.L4,
        "L40S": Machine.L40S,
        "A100": Machine.A100,
        "H100": Machine.H100,
    }
    gpu_machine = machine_map.get(machine.upper(), Machine.L4)

    print("\n" + "=" * 60)
    print("Chess LLM Training - Lightning.ai Runner")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data: {data_file}")
    print(f"  GPU: {machine}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA rank: {lora_rank}")

    print(f"\n[1/6] Creating Lightning Studio: {studio_name}")
    print(f"       Teamspace: {teamspace}")

    try:
        studio = Studio(name=studio_name, teamspace=teamspace, user=username, create_ok=True)
    except Exception as e:
        print(f"Error creating studio: {e}")
        return None

    print(f"\n[2/6] Starting {machine} GPU instance...")
    try:
        studio.start(gpu_machine)
        print(f"  {machine} GPU started!")
    except Exception as e:
        print(f"Error starting GPU: {e}")
        return None

    results = {}

    try:
        # Install dependencies
        print(f"\n[3/6] Installing dependencies...")
        install_cmds = [
            "pip install --upgrade pip -q",
            "pip install torch --index-url https://download.pytorch.org/whl/cu121 -q",
            "pip install transformers datasets accelerate peft trl bitsandbytes -q",
            "pip install python-chess -q",
        ]
        for cmd in install_cmds:
            print(f"  Running: {cmd[:50]}...")
            studio.run_with_exit_code(cmd)

        # Upload training data
        print(f"\n[4/6] Uploading training data...")
        remote_data_dir = "~/chess-training/data"
        remote_data_path = f"{remote_data_dir}/{data_path.name}"

        studio.run_with_exit_code(f"mkdir -p {remote_data_dir}")
        studio.upload_file(str(data_path), remote_data_path)
        print(f"  Uploaded: {data_path.name}")

        # Generate and run training script
        print(f"\n[5/6] Running training...")
        output_dir = "~/chess-training/models"
        training_script = get_training_script(
            data_path=remote_data_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
        )

        # Save script to studio and run
        studio.run_with_exit_code(f'cat > ~/train_chess.py << \'ENDSCRIPT\'\n{training_script}\nENDSCRIPT')

        start_time = time.time()
        exit_code = studio.run_with_exit_code("python ~/train_chess.py")
        elapsed = time.time() - start_time

        results["training_time"] = elapsed
        results["exit_code"] = exit_code

        if exit_code == 0:
            print(f"\n  Training completed in {elapsed/60:.1f} minutes!")

            # Download results
            print(f"\n[6/6] Downloading trained model...")
            local_output = RESULTS_DIR / f"model_{studio_name}"
            local_output.mkdir(parents=True, exist_ok=True)

            # Download the final model
            try:
                studio.download_file(f"{output_dir}/final", str(local_output))
                results["model_path"] = str(local_output)
                print(f"  Model saved to: {local_output}")
            except Exception as e:
                print(f"  Warning: Could not download model: {e}")
        else:
            print(f"\n  Training failed with exit code: {exit_code}")

    except Exception as e:
        print(f"Error during training: {e}")
        results["error"] = str(e)
    finally:
        # Stop the studio to save costs
        print("\nStopping studio...")
        try:
            studio.stop()
            print("  Studio stopped.")
        except:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Chess LLM training on Lightning.ai")
    parser.add_argument("--data", "-d", required=True, help="Path to training data JSONL")
    parser.add_argument("--studio", "-s", default="chess-llm-training", help="Studio name")
    parser.add_argument("--machine", "-m", default="L4", choices=["T4", "L4", "L40S", "A100", "H100"], help="GPU type")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")

    args = parser.parse_args()

    results = run_training_on_lightning(
        data_file=args.data,
        studio_name=args.studio,
        machine=args.machine,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
    )

    if results:
        print("\n" + "=" * 60)
        print("Results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
