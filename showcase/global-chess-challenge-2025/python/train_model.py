"""
Chess LLM Fine-Tuning with LoRA

Fine-tunes a base model (Qwen2.5-3B) on chess training data using LoRA.

Usage:
    python train_model.py --data data/training.jsonl --output models/chess_llm

Requirements:
    pip install transformers peft accelerate bitsandbytes datasets
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
from datasets import load_dataset, Dataset
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


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    base_model: str = "Qwen/Qwen2.5-3B"
    output_dir: str = "models/chess_llm"

    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # Training parameters
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 512

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # Hardware
    fp16: bool = True
    bf16: bool = False


def load_training_data(data_path: str, tokenizer, max_length: int = 512) -> Dataset:
    """Load and tokenize training data"""
    print(f"Loading training data from: {data_path}")

    # Load JSONL
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Format as chat
    def format_chat(example):
        messages = example["messages"]
        # Format as a single text for causal LM
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}\n"
            elif role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}\n"
        return {"text": text}

    # Convert to dataset
    dataset = Dataset.from_list([format_chat(ex) for ex in examples])

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

    tokenized = dataset.map(tokenize, remove_columns=["text"])
    print(f"Tokenized {len(tokenized)} examples")

    return tokenized


def setup_model_and_tokenizer(config: TrainingConfig):
    """Load base model with quantization and setup LoRA"""
    print(f"Loading base model: {config.base_model}")

    # Quantization config
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.bfloat16,
    )

    # Prepare for k-bit training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    data_path: str,
    config: TrainingConfig,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """Main training function"""
    print("=" * 60)
    print("CHESS LLM TRAINING")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load data
    train_dataset = load_training_data(data_path, tokenizer, config.max_seq_length)

    # Split into train/eval
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    print(f"Train size: {len(train_data)}")
    print(f"Eval size: {len(eval_data)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,
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
    print("\nStarting training...")
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final model
    final_path = Path(config.output_dir) / "final"
    print(f"\nSaving final model to: {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save training config
    config_path = final_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "base_model": config.base_model,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "data_path": data_path,
        }, f, indent=2)

    # Training results
    results = {
        "final_loss": trainer.state.best_metric,
        "total_steps": trainer.state.global_step,
        "model_path": str(final_path),
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best eval loss: {results['final_loss']:.4f}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Model saved to: {results['model_path']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune chess LLM with LoRA")
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument("--output", default="models/chess_llm", help="Output directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B", help="Base model")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--resume", help="Resume from checkpoint")

    args = parser.parse_args()

    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    train(args.data, config, resume_from=args.resume)


if __name__ == "__main__":
    main()
