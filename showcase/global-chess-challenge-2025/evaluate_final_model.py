#!/usr/bin/env python3
"""
Evaluate final chess LoRA model using ACPL metric.

Uses forced <uci_move> prefix for correct output format.
"""
import os
import sys
import json
import torch
import re
from pathlib import Path
from typing import List

# Add python dir to path for imports
sys.path.insert(0, str(Path(__file__).parent / "python"))

from evaluate import (
    ChessModelEvaluator,
    load_positions_from_training_data,
    generate_test_positions_random
)

# Model configuration
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_LORA_PATH = Path(__file__).parent / "models" / "final"


def load_model(lora_path: Path = None, base_model: str = None, device: str = "auto"):
    """Load base model with LoRA adapter"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if lora_path is None:
        lora_path = DEFAULT_LORA_PATH
    if base_model is None:
        base_model = DEFAULT_BASE_MODEL

    print(f"Loading base model: {base_model}")
    print(f"LoRA adapter: {lora_path}")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Load tokenizer from base model (has chat template)
    # LoRA checkpoints may not preserve the chat template
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map={"": device} if device in ["mps", "cpu"] else "auto",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()

    return model, tokenizer, device


def create_chess_prompt(fen: str, legal_moves: List[str]) -> List[dict]:
    """Create chat prompt for chess move prediction"""
    moves_str = ", ".join(legal_moves)

    return [
        {
            "role": "system",
            "content": "You are a chess grandmaster. Analyze positions and select the best move."
        },
        {
            "role": "user",
            "content": f"FEN: {fen}\nMoves: {moves_str}\nBest?"
        }
    ]


def make_chess_model_fn(model, tokenizer, device):
    """Create model function for evaluator with forced <uci_move> prefix"""

    def model_fn(fen: str, legal_moves: List[str]) -> str:
        # Create prompt
        messages = create_chess_prompt(fen, legal_moves)

        # Tokenize with generation prompt
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # CRITICAL: Force the model to start with <uci_move>
        text += "<uci_move>"

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # Greedy for evaluation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens, prepend the forced prefix
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = "<uci_move>" + tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    return model_fn


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate final chess model")
    parser.add_argument("--model-path", type=str, default=None, help="Path to LoRA adapter (default: models/final)")
    parser.add_argument("--base-model", type=str, default=None, help="Base model (default: Qwen/Qwen2.5-3B, use Qwen/Qwen2.5-7B for 7B)")
    parser.add_argument("--positions", type=int, default=100, help="Number of positions to evaluate")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish depth")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        lora_path = Path(args.model_path)
        if not lora_path.is_absolute():
            lora_path = Path(__file__).parent / args.model_path
    else:
        lora_path = DEFAULT_LORA_PATH

    # Auto-detect 7B model from path
    base_model = args.base_model
    if base_model is None and "7b" in str(lora_path).lower():
        base_model = "Qwen/Qwen2.5-7B"
        print(f"Auto-detected 7B model from path")

    print("=" * 60)
    print("Chess Final Model Evaluation")
    print("=" * 60)
    print()

    # Load model
    model, tokenizer, device = load_model(lora_path, base_model, args.device)
    print()

    # Create model function
    model_fn = make_chess_model_fn(model, tokenizer, device)

    # Test single prediction
    print("Testing single prediction...")
    test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    test_moves = ["a7a6", "b7b6", "c7c6", "d7d6", "e7e6", "f7f6", "g7g6", "h7h6",
                  "a7a5", "b7b5", "c7c5", "d7d5", "e7e5", "f7f5", "g7g5", "h7h5",
                  "b8a6", "b8c6", "g8f6", "g8h6"]
    response = model_fn(test_fen, test_moves)
    print(f"  FEN: {test_fen}")
    print(f"  Response: {response[:100]}...")

    # Extract move for verification
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
    if match:
        print(f"  Move: {match.group(1)} ({'LEGAL' if match.group(1) in test_moves else 'ILLEGAL'})")
    else:
        print("  WARNING: No valid move extracted!")
    print()

    # Load positions
    training_data = Path(__file__).parent / "data" / "training_elite_30k.jsonl"

    if training_data.exists():
        print(f"Loading {args.positions} positions from training data...")
        # Use different seed than training to get diverse positions
        positions = load_positions_from_training_data(str(training_data), n=args.positions, seed=12345)
        position_source = "training_elite_30k.jsonl (seed=12345)"
    else:
        print("WARNING: Using random positions")
        positions = generate_test_positions_random(n=args.positions, seed=42)
        position_source = "random_walk"

    print(f"Loaded {len(positions)} positions")
    print()

    # Run evaluation
    print(f"Running ACPL evaluation (depth={args.depth})...")
    print("This may take a while...")
    print()

    evaluator = ChessModelEvaluator(model_fn)
    results = evaluator.evaluate(
        positions,
        depth=args.depth,
        strict_parsing=True,
        position_source=position_source
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  ACPL: {results['acpl']:.1f}")
    print(f"  ACPL (legal only): {results['acpl_legal_only']:.1f}")
    print(f"  Legal Move Rate: {results['legal_rate']*100:.1f}%")
    print(f"  Best Move Rate: {results['best_move_rate']*100:.1f}%")
    print(f"  Positions: {results['n_positions']} ({results['n_legal']} legal, {results['n_illegal']} illegal)")
    print()
    print("Config:")
    print(f"  Model: {base_model or DEFAULT_BASE_MODEL} + LoRA")
    print(f"  LoRA: {lora_path}")
    print(f"  Device: {device}")
    print(f"  Stockfish: {results['config']['stockfish_version']}")
    print(f"  Depth: {results['config']['depth']}")

    # Gating check
    print()
    if results['legal_rate'] < 0.995:
        print("WARNING: Legal rate < 99.5% - model may have output format issues")
    else:
        print("Legal rate OK (>= 99.5%)")

    # Competition context
    print()
    print("Competition Reference:")
    print("  - Random baseline ACPL: ~400-500")
    print("  - Good amateur (1500 Elo): ~150-200 ACPL")
    print("  - Strong club (2000 Elo): ~80-120 ACPL")
    print("  - Master (2200+ Elo): ~50-80 ACPL")
    print("  - Grandmaster (2500+): ~30-50 ACPL")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "eval_final_model.json"

    output_path.parent.mkdir(exist_ok=True)

    # Remove non-serializable results for saving
    save_results = {k: v for k, v in results.items() if k != 'results'}
    save_results['model'] = {
        'base': base_model or DEFAULT_BASE_MODEL,
        'lora_path': str(lora_path),
        'device': device
    }

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
