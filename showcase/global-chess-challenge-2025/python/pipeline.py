"""
Complete Training Pipeline for Chess Challenge

Orchestrates the full workflow:
1. Extract training data from PGN
2. Fine-tune model with LoRA
3. Evaluate against Stockfish
4. Update evolution state

Usage:
    python pipeline.py --config .evolve-sdk/evolve_chess_training_strategy/mutations/gen3_cross.json
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))

from extract_training_data import extract_training_data, DataConfig
from train_model import train, TrainingConfig


def load_evolution_config(config_path: str) -> Dict[str, Any]:
    """Load an evolution candidate config"""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_pipeline(
    config_path: str,
    pgn_path: str,
    output_dir: str,
    skip_extract: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    max_examples: int = 5000,
    eval_positions: int = 50,
) -> Dict[str, Any]:
    """
    Run the complete training pipeline.

    Steps:
    1. Extract training data using config
    2. Train model with LoRA
    3. Evaluate against Stockfish
    """
    print("=" * 70)
    print("CHESS LLM TRAINING PIPELINE")
    print("=" * 70)

    # Load config
    config = load_evolution_config(config_path)
    config_id = config.get("id", "unknown")

    print(f"\nConfiguration: {config_id}")
    print(f"Description: {config.get('description', 'N/A')}")

    output_dir = Path(output_dir) / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config_id": config_id,
        "timestamp": datetime.now().isoformat(),
        "stages": {},
    }

    # Stage 1: Data Extraction
    if not skip_extract:
        print("\n" + "=" * 70)
        print("STAGE 1: DATA EXTRACTION")
        print("=" * 70)

        data_config = config.get("data_config", {})
        extraction_config = DataConfig(
            min_elo=data_config.get("min_elo", 2200),
            max_elo=data_config.get("max_elo", 2800),
            min_move_number=data_config.get("min_move_number", 10),
            max_move_number=data_config.get("max_move_number", 60),
            opening_fraction=data_config.get("opening_fraction", 0.2),
            middlegame_fraction=data_config.get("middlegame_fraction", 0.6),
            endgame_fraction=data_config.get("endgame_fraction", 0.2),
            stockfish_depth=data_config.get("stockfish_depth", 18),
            max_examples=max_examples,
        )

        training_data_path = output_dir / "training_data.jsonl"

        prompt_style = config.get("prompt_format", {}).get("style", "concise")

        extraction_stats = extract_training_data(
            pgn_path,
            str(training_data_path),
            extraction_config,
            prompt_style=prompt_style,
        )

        results["stages"]["extraction"] = {
            "status": "success",
            "examples": extraction_stats["total_examples"],
            "output": str(training_data_path),
        }
    else:
        training_data_path = output_dir / "training_data.jsonl"
        print(f"\nSkipping extraction, using: {training_data_path}")

    # Stage 2: Model Training
    if not skip_train:
        print("\n" + "=" * 70)
        print("STAGE 2: MODEL TRAINING")
        print("=" * 70)

        train_config = config.get("training_config", {})
        training_config = TrainingConfig(
            base_model=train_config.get("base_model", "Qwen/Qwen2.5-3B"),
            output_dir=str(output_dir / "model"),
            lora_rank=train_config.get("lora_rank", 64),
            lora_alpha=train_config.get("lora_alpha", 128),
            learning_rate=train_config.get("learning_rate", 1e-5),
            num_epochs=train_config.get("num_epochs", 3),
            batch_size=train_config.get("batch_size", 4),
        )

        train_results = train(str(training_data_path), training_config)

        results["stages"]["training"] = {
            "status": "success",
            "final_loss": train_results["final_loss"],
            "model_path": train_results["model_path"],
        }

        model_path = train_results["model_path"]
    else:
        model_path = str(output_dir / "model" / "final")
        print(f"\nSkipping training, using: {model_path}")

    # Stage 3: Evaluation
    if not skip_eval:
        print("\n" + "=" * 70)
        print("STAGE 3: EVALUATION")
        print("=" * 70)

        # Import here to avoid loading model twice
        from evaluate_model import ChessModelEvaluator, generate_test_positions

        evaluator = ChessModelEvaluator(model_path)
        positions = generate_test_positions(eval_positions, seed=42)

        eval_results = evaluator.evaluate_dataset(positions, depth=18)
        evaluator.close()

        results["stages"]["evaluation"] = {
            "status": "success",
            "acpl": eval_results["acpl"],
            "legal_rate": eval_results["legal_rate"],
            "best_move_rate": eval_results["best_move_rate"],
        }

        # Update config with actual fitness
        results["fitness"] = eval_results["acpl"]
    else:
        print("\nSkipping evaluation")

    # Save results
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Config: {config_id}")
    print(f"Output: {output_dir}")

    if "evaluation" in results["stages"]:
        eval_stage = results["stages"]["evaluation"]
        print(f"\nFinal Metrics:")
        print(f"  ACPL: {eval_stage['acpl']:.1f}")
        print(f"  Legal Rate: {eval_stage['legal_rate']*100:.1f}%")
        print(f"  Best Move Rate: {eval_stage['best_move_rate']*100:.1f}%")

    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run chess training pipeline")
    parser.add_argument("--config", required=True, help="Path to evolution config JSON")
    parser.add_argument("--pgn", default="data/games/lichess_db_standard_rated_2013-01.pgn",
                       help="Path to PGN file")
    parser.add_argument("--output", default="runs", help="Output directory")
    parser.add_argument("--max-examples", type=int, default=5000, help="Max training examples")
    parser.add_argument("--eval-positions", type=int, default=50, help="Evaluation positions")
    parser.add_argument("--skip-extract", action="store_true", help="Skip data extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")

    args = parser.parse_args()

    results = run_pipeline(
        args.config,
        args.pgn,
        args.output,
        skip_extract=args.skip_extract,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        max_examples=args.max_examples,
        eval_positions=args.eval_positions,
    )

    # Exit with success if ACPL < 200
    if results.get("fitness", 999) < 200:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
