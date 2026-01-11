#!/usr/bin/env python3
"""
Evolutionary Training Data Optimization

Evolves training data configurations to minimize ACPL.
Each generation:
1. Generate data with different filtering criteria
2. Train on Lightning.ai (~$15-20 per run)
3. Evaluate ACPL
4. Select best performers
5. Mutate/crossover for next generation

Budget: ~$200-300 = 10-15 training runs
Strategy: 2-3 generations of 4-5 population

Usage:
    python evolve_training.py --generations 3 --population 4
    python evolve_training.py --resume
"""
import os
import sys
import json
import random
import copy
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from data_config import DataConfig, BASELINE, generate_initial_population

# Load environment
load_dotenv()

LIGHTNING_USERNAME = os.getenv("LIGHTNING_AI_USERNAME")
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE")


def generate_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


class TrainingEvolution:
    """Manages the evolution of training configurations"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent
        self.state_path = self.output_dir / "evolution_state.json"
        self.configs_dir = self.output_dir / "configs"
        self.results_dir = self.output_dir / "results"

        self.configs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        self.state = {
            "generation": 0,
            "population": [],
            "champion": None,
            "history": [],
            "total_cost": 0.0,
            "start_time": None,
        }

    def save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def load_state(self):
        if self.state_path.exists():
            with open(self.state_path) as f:
                self.state = json.load(f)

    def extract_training_data(self, config: DataConfig, config_id: str) -> Path:
        """
        Extract training data based on config.
        Returns path to JSONL file.
        """
        from python.extract_training_data import extract_training_data

        output_path = self.configs_dir / f"{config_id}_training.jsonl"

        print(f"  Extracting training data for {config_id}...")
        print(f"    Elo >= {config.min_elo}")
        print(f"    SF top-{config.stockfish_top_n}" if config.stockfish_top_n > 0 else "    No SF filter")
        print(f"    CPL <= {config.max_centipawn_loss}")

        # This would call the actual extraction function
        # For now, we'll use the existing training data as base
        # and filter it according to config

        # TODO: Implement actual data extraction with filters
        # For initial version, copy baseline data
        base_data = Path(__file__).parent.parent.parent / "data" / "training_elite_30k.jsonl"
        if base_data.exists():
            import shutil
            shutil.copy(base_data, output_path)

        return output_path

    def run_training(self, config: DataConfig, config_id: str, data_path: Path) -> dict:
        """
        Run training on Lightning.ai.
        Returns training results including checkpoint path.
        """
        os.environ["LIGHTNING_USER_ID"] = os.getenv("LIGHTNING_USER_ID")
        os.environ["LIGHTNING_API_KEY"] = os.getenv("LIGHTNING_API_KEY")

        from lightning_sdk import Studio

        print(f"\n  Starting training for {config_id} on Lightning.ai...")

        studio = Studio(
            name="chess-llm-v4",
            teamspace=LIGHTNING_TEAMSPACE,
            user=LIGHTNING_USERNAME,
        )

        # Start studio if not running
        if studio.status != "Running":
            print("  Starting studio...")
            studio.start()
            time.sleep(10)

        # Upload training data
        remote_data_path = f"chess-training/evolution/{config_id}/training.jsonl"
        print(f"  Uploading training data to {remote_data_path}...")
        studio.upload_file(str(data_path), remote_data_path)

        # Generate training script for this config
        train_script = f'''
#!/usr/bin/env python3
"""Training script for config {config_id}"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_PATH = Path.home() / "{remote_data_path}"
OUTPUT_DIR = Path.home() / "chess-training" / "evolution" / "{config_id}" / "model"
MODEL_NAME = "Qwen/Qwen2.5-3B"

# Config from evolution
EPOCHS = {config.epochs}
BATCH_SIZE = {config.batch_size}
LEARNING_RATE = {config.learning_rate}
LORA_RANK = {config.lora_rank}
LORA_ALPHA = {config.lora_alpha}

print("Loading data...")
with open(DATA_PATH, 'r') as f:
    examples = [json.loads(line) for line in f][:30000]

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
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
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

print(f"Training complete! Model saved to {{OUTPUT_DIR / 'final'}}")
'''

        # Upload and run training script
        script_path = self.configs_dir / f"{config_id}_train.py"
        with open(script_path, 'w') as f:
            f.write(train_script)

        studio.upload_file(str(script_path), f"chess-training/evolution/{config_id}/train.py")

        print("  Running training (this will take ~4-6 hours)...")
        result = studio.run(
            f"cd ~/chess-training/evolution/{config_id} && python train.py 2>&1 | tee train.log"
        )

        # Check for success
        if "Training complete!" in result:
            return {
                "success": True,
                "checkpoint_path": f"chess-training/evolution/{config_id}/model/final",
                "log": result[-2000:],  # Last 2000 chars
            }
        else:
            return {
                "success": False,
                "error": result[-2000:],
            }

    def evaluate_model(self, config_id: str, checkpoint_path: str) -> dict:
        """
        Download checkpoint and evaluate ACPL.
        """
        from evaluate_final_model import evaluate_fitness

        print(f"\n  Evaluating {config_id}...")

        # Download checkpoint from Lightning.ai
        local_path = self.results_dir / config_id / "model"
        local_path.mkdir(parents=True, exist_ok=True)

        os.environ["LIGHTNING_USER_ID"] = os.getenv("LIGHTNING_USER_ID")
        os.environ["LIGHTNING_API_KEY"] = os.getenv("LIGHTNING_API_KEY")

        from lightning_sdk import Studio

        studio = Studio(
            name="chess-llm-v4",
            teamspace=LIGHTNING_TEAMSPACE,
            user=LIGHTNING_USERNAME,
        )

        # Download model files
        files = ["adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json"]

        for fname in files:
            try:
                studio.download_file(f"{checkpoint_path}/{fname}", str(local_path / fname))
            except Exception as e:
                print(f"    Warning: Could not download {fname}: {e}")

        # Run ACPL evaluation
        # TODO: Implement proper evaluation using local checkpoint
        # For now, return placeholder
        return {
            "acpl": 999.9,  # Placeholder
            "legal_rate": 0.0,
            "evaluated": False,
        }

    def mutate(self, config: DataConfig) -> DataConfig:
        """Create mutated offspring"""
        new_config = copy.deepcopy(config)

        # Pick random mutation
        mutation_type = random.choice([
            "elo", "stockfish", "cpl", "epochs", "lora_rank", "phases"
        ])

        if mutation_type == "elo":
            delta = random.choice([-200, -100, 100, 200])
            new_config.min_elo = max(2000, min(2800, new_config.min_elo + delta))

        elif mutation_type == "stockfish":
            new_config.stockfish_top_n = random.choice([0, 1, 3, 5, 10])

        elif mutation_type == "cpl":
            new_config.max_centipawn_loss = random.choice([25, 50, 75, 100, 150])

        elif mutation_type == "epochs":
            new_config.epochs = random.choice([2, 3, 4, 5])

        elif mutation_type == "lora_rank":
            new_config.lora_rank = random.choice([32, 64, 128])
            new_config.lora_alpha = new_config.lora_rank * 2

        elif mutation_type == "phases":
            new_config.include_opening = random.choice([True, False]) or True
            new_config.include_endgame = random.choice([True, False])

        return new_config

    def crossover(self, config1: DataConfig, config2: DataConfig) -> DataConfig:
        """Create offspring by combining two parents"""
        return DataConfig(
            min_elo=random.choice([config1.min_elo, config2.min_elo]),
            stockfish_top_n=random.choice([config1.stockfish_top_n, config2.stockfish_top_n]),
            max_centipawn_loss=random.choice([config1.max_centipawn_loss, config2.max_centipawn_loss]),
            epochs=random.choice([config1.epochs, config2.epochs]),
            lora_rank=random.choice([config1.lora_rank, config2.lora_rank]),
            lora_alpha=random.choice([config1.lora_alpha, config2.lora_alpha]),
            include_opening=random.choice([config1.include_opening, config2.include_opening]),
            include_middlegame=True,
            include_endgame=random.choice([config1.include_endgame, config2.include_endgame]),
        )

    def run_evolution(
        self,
        generations: int = 3,
        population_size: int = 4,
        resume: bool = False,
    ):
        """Run the full evolution loop"""

        if resume and self.state_path.exists():
            print("Resuming from checkpoint...")
            self.load_state()
            start_gen = self.state["generation"] + 1
        else:
            print("Initializing evolution...")
            self.state["start_time"] = datetime.now().isoformat()

            # Generate initial population
            initial_configs = generate_initial_population(population_size)
            self.state["population"] = [
                {
                    "id": generate_id() if i > 0 else "baseline",
                    "generation": 0,
                    "config": asdict(config),
                    "fitness": None,
                    "status": "pending",
                }
                for i, config in enumerate(initial_configs)
            ]
            self.save_state()
            start_gen = 0

        print(f"\nStarting evolution from generation {start_gen}")
        print(f"Population size: {population_size}")
        print(f"Estimated cost: ${population_size * generations * 15:.0f}-${population_size * generations * 20:.0f}")
        print()

        for gen in range(start_gen, generations + 1):
            print(f"\n{'='*60}")
            print(f"GENERATION {gen}")
            print(f"{'='*60}")

            # Process pending individuals
            for ind in self.state["population"]:
                if ind["status"] == "pending":
                    config = DataConfig.from_dict(ind["config"])
                    config_id = ind["id"]

                    print(f"\n[{config_id}] {config.get_name()}")

                    # Extract data
                    data_path = self.extract_training_data(config, config_id)

                    # Train
                    train_result = self.run_training(config, config_id, data_path)

                    if train_result["success"]:
                        # Evaluate
                        eval_result = self.evaluate_model(config_id, train_result["checkpoint_path"])
                        ind["fitness"] = eval_result
                        ind["status"] = "evaluated"
                        print(f"  ACPL: {eval_result['acpl']:.1f}")
                    else:
                        ind["status"] = "failed"
                        ind["fitness"] = {"acpl": 10000, "error": train_result.get("error", "Unknown")}
                        print(f"  FAILED: {train_result.get('error', 'Unknown')[:100]}")

                    self.state["total_cost"] += 17.5  # Estimated cost per run
                    self.save_state()

            # Update champion
            evaluated = [ind for ind in self.state["population"] if ind.get("fitness")]
            if evaluated:
                best = min(evaluated, key=lambda x: x["fitness"].get("acpl", 10000))
                if self.state["champion"] is None or best["fitness"]["acpl"] < self.state["champion"]["fitness"]["acpl"]:
                    self.state["champion"] = best
                    print(f"\n*** NEW CHAMPION: {best['id']} (ACPL: {best['fitness']['acpl']:.1f}) ***")

            # Record history
            acpls = [ind["fitness"]["acpl"] for ind in evaluated if ind["fitness"]]
            self.state["history"].append({
                "generation": gen,
                "best_acpl": min(acpls) if acpls else 10000,
                "avg_acpl": sum(acpls) / len(acpls) if acpls else 10000,
                "population_size": len(self.state["population"]),
            })

            # Generate next generation (if not last)
            if gen < generations:
                print(f"\nGenerating generation {gen + 1}...")

                # Elitism: keep top 2
                evaluated.sort(key=lambda x: x["fitness"].get("acpl", 10000))
                new_population = evaluated[:2]

                # Generate offspring
                while len(new_population) < population_size:
                    if random.random() < 0.7:
                        # Mutation
                        parent = random.choice(evaluated[:3])  # Top 3
                        parent_config = DataConfig.from_dict(parent["config"])
                        child_config = self.mutate(parent_config)
                    else:
                        # Crossover
                        p1, p2 = random.sample(evaluated[:3], 2)
                        c1 = DataConfig.from_dict(p1["config"])
                        c2 = DataConfig.from_dict(p2["config"])
                        child_config = self.crossover(c1, c2)

                    new_population.append({
                        "id": generate_id(),
                        "generation": gen + 1,
                        "config": asdict(child_config),
                        "fitness": None,
                        "status": "pending",
                    })

                self.state["population"] = new_population
                self.state["generation"] = gen

            self.save_state()

        # Final summary
        print("\n" + "="*60)
        print("EVOLUTION COMPLETE")
        print("="*60)
        print(f"Total cost: ~${self.state['total_cost']:.0f}")
        print(f"Champion: {self.state['champion']['id']}")
        print(f"Champion ACPL: {self.state['champion']['fitness']['acpl']:.1f}")
        print(f"Champion config: {DataConfig.from_dict(self.state['champion']['config']).get_name()}")


def main():
    parser = argparse.ArgumentParser(description="Evolve training configurations")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations")
    parser.add_argument("--population", type=int, default=4, help="Population size")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    args = parser.parse_args()

    evolution = TrainingEvolution()

    if args.dry_run:
        print("DRY RUN - Evolution Plan")
        print("="*60)
        print(f"Generations: {args.generations}")
        print(f"Population: {args.population}")
        print(f"Total training runs: {args.generations * args.population}")
        print(f"Estimated cost: ${args.generations * args.population * 15:.0f}-${args.generations * args.population * 20:.0f}")
        print(f"Estimated time: {args.generations * args.population * 5:.0f}-{args.generations * args.population * 7:.0f} hours")
        print()
        print("Initial population:")
        for i, config in enumerate(generate_initial_population(args.population)):
            print(f"  {i+1}. {config.get_name()}")
        return

    evolution.run_evolution(
        generations=args.generations,
        population_size=args.population,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
