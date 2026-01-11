#!/usr/bin/env python3
"""
Fitness Evaluator for Chess Prompt Evolution

Measures ACPL (Average Centipawn Loss) for a given prompt configuration.
Lower ACPL = better fitness.

Usage:
    python fitness_evaluator.py <config.json> [--positions N] [--depth D]
"""
import sys
import json
import torch
import re
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from prompt_template import PromptConfig, FewShotExample

# Lazy imports for speed
_model = None
_tokenizer = None
_device = None
_evaluator = None
_positions = None


def get_model():
    """Lazy load model (expensive, do once)"""
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    BASE_MODEL = "Qwen/Qwen2.5-3B"
    LORA_PATH = Path(__file__).parent.parent.parent / "models" / "final"

    # Device
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    print(f"Loading model on {_device}...", file=sys.stderr)

    _tokenizer = AutoTokenizer.from_pretrained(str(LORA_PATH))

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if _device != "cpu" else torch.float32,
        device_map={"": _device} if _device in ["mps", "cpu"] else "auto",
        trust_remote_code=True
    )
    _model = PeftModel.from_pretrained(_model, str(LORA_PATH))
    _model.eval()

    print("Model loaded.", file=sys.stderr)
    return _model, _tokenizer, _device


def get_evaluator():
    """Lazy load Stockfish evaluator"""
    global _evaluator

    if _evaluator is not None:
        return _evaluator

    from evaluate import ChessModelEvaluator

    # Dummy model function - we'll override per-call
    def dummy_fn(fen, moves):
        return ""

    _evaluator = ChessModelEvaluator(dummy_fn)
    return _evaluator


def get_positions(n: int = 50, seed: int = 99999):
    """Get test positions (cached)"""
    global _positions

    if _positions is not None and len(_positions) >= n:
        return _positions[:n]

    from evaluate import load_positions_from_training_data

    data_path = Path(__file__).parent.parent.parent / "data" / "training_elite_30k.jsonl"
    _positions = load_positions_from_training_data(str(data_path), n=max(n, 100), seed=seed)

    return _positions[:n]


def build_messages(config: PromptConfig, fen: str, legal_moves: List[str]) -> List[dict]:
    """Build chat messages from config"""
    messages = []

    # System message
    messages.append({
        "role": "system",
        "content": config.system_prompt
    })

    # Few-shot examples
    for ex in config.few_shot_examples:
        # User turn
        user_content = config.user_template.format(
            fen=ex.fen,
            moves=", ".join(ex.moves)
        )
        messages.append({"role": "user", "content": user_content})

        # Assistant turn (the answer)
        assistant_content = f"<uci_move>{ex.best_move}</uci_move>\n<rationale>{ex.rationale}</rationale>"
        messages.append({"role": "assistant", "content": assistant_content})

    # Current position (user turn)
    user_content = config.user_template.format(
        fen=fen,
        moves=", ".join(legal_moves)
    )
    if config.format_hint:
        user_content += config.format_hint

    messages.append({"role": "user", "content": user_content})

    return messages


def make_model_fn(config: PromptConfig):
    """Create model function for a specific prompt config"""
    model, tokenizer, device = get_model()

    def model_fn(fen: str, legal_moves: List[str]) -> str:
        messages = build_messages(config, fen, legal_moves)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Add forced prefix
        if config.force_prefix:
            text += config.force_prefix

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=config.temperature if config.temperature > 0 else None,
                top_p=config.top_p if config.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = config.force_prefix + tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    return model_fn


@dataclass
class FitnessResult:
    """Result of fitness evaluation"""
    acpl: float  # Primary fitness (lower = better)
    acpl_legal_only: float
    legal_rate: float
    best_move_rate: float
    n_positions: int
    n_legal: int
    n_illegal: int

    def to_dict(self):
        return {
            "acpl": self.acpl,
            "acpl_legal_only": self.acpl_legal_only,
            "legal_rate": self.legal_rate,
            "best_move_rate": self.best_move_rate,
            "n_positions": self.n_positions,
            "n_legal": self.n_legal,
            "n_illegal": self.n_illegal,
        }


def evaluate_fitness(
    config: PromptConfig,
    n_positions: int = 30,
    depth: int = 12,
    verbose: bool = True
) -> FitnessResult:
    """
    Evaluate fitness of a prompt configuration.

    Returns FitnessResult with ACPL as primary metric (lower = better).
    """
    # Get components
    model_fn = make_model_fn(config)
    evaluator = get_evaluator()
    positions = get_positions(n_positions)

    # Override evaluator's model function
    evaluator.model_fn = model_fn

    if verbose:
        print(f"Evaluating on {len(positions)} positions (depth={depth})...", file=sys.stderr)

    # Run evaluation
    results = evaluator.evaluate(
        positions,
        depth=depth,
        strict_parsing=True,
        position_source="evolution_test"
    )

    return FitnessResult(
        acpl=results["acpl"],
        acpl_legal_only=results["acpl_legal_only"],
        legal_rate=results["legal_rate"],
        best_move_rate=results["best_move_rate"],
        n_positions=results["n_positions"],
        n_legal=results["n_legal"],
        n_illegal=results["n_illegal"],
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess prompt fitness")
    parser.add_argument("config", type=str, nargs="?", help="Path to prompt config JSON")
    parser.add_argument("--positions", type=int, default=30, help="Number of positions")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish depth")
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline config")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    # Load config
    if args.baseline or not args.config:
        from prompt_template import BASELINE
        config = BASELINE
        config_name = "baseline"
    else:
        config = PromptConfig.load(args.config)
        config_name = Path(args.config).stem

    if not args.json:
        print(f"Config: {config_name}")
        print(f"System: {config.system_prompt[:50]}...")
        print(f"Few-shots: {len(config.few_shot_examples)}")
        print()

    # Evaluate
    result = evaluate_fitness(
        config,
        n_positions=args.positions,
        depth=args.depth,
        verbose=not args.json
    )

    if args.json:
        output = {
            "config": config_name,
            "fitness": result.to_dict(),
        }
        print(json.dumps(output))
    else:
        print()
        print("=" * 50)
        print("FITNESS RESULTS")
        print("=" * 50)
        print(f"  ACPL (fitness):    {result.acpl:.1f}")
        print(f"  ACPL (legal only): {result.acpl_legal_only:.1f}")
        print(f"  Legal rate:        {result.legal_rate*100:.1f}%")
        print(f"  Best move rate:    {result.best_move_rate*100:.1f}%")
        print(f"  Positions:         {result.n_positions} ({result.n_legal} legal)")
        print()
        print(f"Fitness score: {result.acpl:.1f} (lower is better)")


if __name__ == "__main__":
    main()
