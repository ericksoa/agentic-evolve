"""
Chess Model Evaluation Pipeline

Evaluates a trained chess LLM against Stockfish:
1. Load trained model
2. Generate moves for test positions
3. Compute ACPL (Average Centipawn Loss)

Usage:
    python evaluate_model.py --model models/chess_llm/final --positions 100
"""
import argparse
import chess
import chess.engine
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Stockfish configuration
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


@dataclass
class EvalResult:
    """Result for a single position"""
    fen: str
    model_move: str
    best_move: str
    model_cp: int
    best_cp: int
    cp_loss: int
    is_legal: bool
    is_best: bool
    latency_ms: float


class ChessModelEvaluator:
    """Evaluates a chess model against Stockfish"""

    def __init__(
        self,
        model_path: str,
        stockfish_path: str = STOCKFISH_PATH,
        device: str = "auto"
    ):
        self.stockfish_path = stockfish_path
        self.device = device

        print(f"Loading model from: {model_path}")
        self._load_model(model_path)

        print(f"Initializing Stockfish: {stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def _load_model(self, model_path: str):
        """Load the trained model"""
        model_path = Path(model_path)

        # Check if this is a LoRA model
        if (model_path / "adapter_config.json").exists():
            # Load base model info
            with open(model_path / "adapter_config.json") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B")

            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def format_prompt(self, board: chess.Board) -> str:
        """Format position as model prompt"""
        fen = board.fen()
        legal_moves = ", ".join(m.uci() for m in board.legal_moves)

        prompt = f"""<|system|>
You are a chess grandmaster. Analyze positions and select the best move.
<|user|>
FEN: {fen}
Moves: {legal_moves}
Best?
<|assistant|>
"""
        return prompt

    def extract_move(self, response: str) -> Optional[str]:
        """Extract UCI move from model response"""
        # Try to find <uci_move> tag
        match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
        if match:
            return match.group(1)

        # Fallback: find any UCI-like pattern
        match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', response)
        if match:
            return match.group(1)

        return None

    def generate_move(self, board: chess.Board, max_new_tokens: int = 100) -> Tuple[str, float]:
        """Generate a move for the given position"""
        prompt = self.format_prompt(board)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for determinism
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        move = self.extract_move(response)
        return move or "0000", latency_ms

    def analyze_position(self, board: chess.Board, depth: int = 20) -> Tuple[chess.Move, int]:
        """Analyze position with Stockfish"""
        result = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = result["pv"][0]
        score = result["score"].white()

        if score.is_mate():
            cp = 10000 if score.mate() > 0 else -10000
        else:
            cp = score.score()

        return best_move, cp

    def evaluate_position(
        self,
        board: chess.Board,
        depth: int = 20
    ) -> EvalResult:
        """Evaluate model on a single position"""
        fen = board.fen()

        # Get model's move
        model_move_uci, latency_ms = self.generate_move(board)

        # Get Stockfish's best move
        best_move, best_cp = self.analyze_position(board, depth)

        # Check if model's move is legal
        try:
            model_move = chess.Move.from_uci(model_move_uci)
            is_legal = model_move in board.legal_moves
        except ValueError:
            is_legal = False
            model_move = None

        # Evaluate model's move
        if is_legal:
            board.push(model_move)
            _, model_eval = self.analyze_position(board, depth)
            board.pop()

            # Adjust for side to move
            if board.turn == chess.BLACK:
                model_eval = -model_eval
                best_cp = -best_cp

            model_cp = model_eval
        else:
            # Illegal move = worst possible
            model_cp = -10000 if board.turn == chess.WHITE else 10000

        # Calculate centipawn loss
        cp_loss = max(0, best_cp - model_cp)

        return EvalResult(
            fen=fen,
            model_move=model_move_uci,
            best_move=best_move.uci(),
            model_cp=model_cp,
            best_cp=best_cp,
            cp_loss=cp_loss,
            is_legal=is_legal,
            is_best=is_legal and model_move == best_move,
            latency_ms=latency_ms,
        )

    def evaluate_dataset(
        self,
        positions: List[chess.Board],
        depth: int = 20
    ) -> Dict[str, Any]:
        """Evaluate model on multiple positions"""
        results = []

        for i, board in enumerate(positions):
            if (i + 1) % 10 == 0:
                print(f"  Evaluating position {i + 1}/{len(positions)}...")

            result = self.evaluate_position(board, depth)
            results.append(result)

        # Compute metrics
        legal_results = [r for r in results if r.is_legal]
        acpl = sum(r.cp_loss for r in results) / len(results) if results else 0
        legal_rate = len(legal_results) / len(results) if results else 0
        best_rate = sum(1 for r in results if r.is_best) / len(results) if results else 0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        return {
            "acpl": acpl,
            "legal_rate": legal_rate,
            "best_move_rate": best_rate,
            "avg_latency_ms": avg_latency,
            "n_positions": len(results),
            "n_legal": len(legal_results),
            "results": [vars(r) for r in results],
        }

    def close(self):
        """Clean up resources"""
        self.engine.quit()


def generate_test_positions(n: int = 100, seed: int = 42) -> List[chess.Board]:
    """Generate diverse test positions"""
    random.seed(seed)
    positions = []

    while len(positions) < n:
        board = chess.Board()

        # Play random moves
        num_moves = random.randint(10, 60)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(random.choice(legal_moves))

        # Skip terminal or trivial positions
        if board.is_game_over():
            continue
        if len(board.piece_map()) < 6:
            continue

        positions.append(board.copy())

    return positions


def load_positions_from_file(filepath: str, max_positions: int = 100) -> List[chess.Board]:
    """Load test positions from a file (one FEN per line)"""
    positions = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    board = chess.Board(line)
                    positions.append(board)
                    if len(positions) >= max_positions:
                        break
                except ValueError:
                    continue
    return positions


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--positions", type=int, default=50, help="Number of test positions")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish analysis depth")
    parser.add_argument("--test-file", help="File with test positions (FEN per line)")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for position generation")

    args = parser.parse_args()

    print("=" * 60)
    print("CHESS MODEL EVALUATION")
    print("=" * 60)

    # Load evaluator
    evaluator = ChessModelEvaluator(args.model)

    # Generate or load test positions
    if args.test_file:
        print(f"Loading positions from: {args.test_file}")
        positions = load_positions_from_file(args.test_file, args.positions)
    else:
        print(f"Generating {args.positions} random test positions...")
        positions = generate_test_positions(args.positions, args.seed)

    print(f"Evaluating on {len(positions)} positions (depth={args.depth})...")

    # Evaluate
    results = evaluator.evaluate_dataset(positions, args.depth)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"ACPL: {results['acpl']:.1f}")
    print(f"Legal Move Rate: {results['legal_rate']*100:.1f}%")
    print(f"Best Move Rate: {results['best_move_rate']*100:.1f}%")
    print(f"Avg Latency: {results['avg_latency_ms']:.1f}ms")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    evaluator.close()

    # Return ACPL for scripting
    return results['acpl']


if __name__ == "__main__":
    acpl = main()
    sys.exit(0 if acpl < 200 else 1)
