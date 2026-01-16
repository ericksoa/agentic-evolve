#!/usr/bin/env python3
"""
Mini RL Validation for Global Chess Challenge 2025

This script validates the RL infrastructure without full gradient updates.
It demonstrates that:
1. We can sample positions from training data
2. We can compute rewards (negative centipawn loss) from Stockfish
3. The reward signal is informative (varies with move quality)

This is a de-risking experiment before committing to expensive H100 training.

Usage:
    python mini_rl_validation.py [--num-positions 100] [--depth 12] [--model-path PATH]
"""

import os
import sys
import json
import random
import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
import statistics

import chess
import chess.engine

# Optional: For model inference (if available)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# Configuration
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class RewardResult:
    """Result of computing reward for a single position"""
    fen: str
    model_move: str
    best_move: str
    model_cp_after: int  # Centipawn score after model move
    best_cp_after: int   # Centipawn score after best move
    cp_loss: int         # How much worse model is vs best
    reward: float        # Negative CP loss (higher is better)
    is_legal: bool
    is_best: bool
    inference_time: float
    analysis_time: float


@dataclass
class ValidationSummary:
    """Summary of RL validation run"""
    timestamp: str
    num_positions: int
    stockfish_depth: int

    # Reward statistics
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float

    # Quality metrics
    legal_rate: float
    best_move_rate: float
    mean_cp_loss: float

    # Timing
    total_time: float
    avg_inference_time: float
    avg_analysis_time: float

    # RL readiness
    reward_signal_informative: bool
    infrastructure_ready: bool
    recommendation: str


def score_to_cp(score: chess.engine.PovScore) -> int:
    """Convert PovScore to centipawns from white's perspective"""
    s = score.white()
    if s.is_mate():
        return 10000 if s.mate() > 0 else -10000
    return s.score(mate_score=10000)


def load_positions_from_training_data(
    jsonl_path: Path,
    n: int = 100,
    seed: int = 42
) -> List[chess.Board]:
    """Load positions from training JSONL file"""
    random.seed(seed)
    all_positions = []

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)

            # Extract text to search for FEN
            text_to_search = ""
            if "messages" in data:
                for msg in data["messages"]:
                    if msg.get("role") == "user":
                        text_to_search = msg.get("content", "")
                        break
            elif "prompt" in data:
                text_to_search = data["prompt"]

            # Extract FEN
            fen_match = re.search(r'FEN:\s*([^\n]+)', text_to_search)
            if fen_match:
                fen = fen_match.group(1).strip()
                try:
                    board = chess.Board(fen)
                    if not board.is_game_over():
                        all_positions.append(board)
                except:
                    pass

    # Sample if we have enough
    if len(all_positions) >= n:
        return random.sample(all_positions, n)
    return all_positions


def generate_random_positions(n: int = 100, seed: int = 42) -> List[chess.Board]:
    """Generate positions via random walks (fallback)"""
    random.seed(seed)
    positions = []

    while len(positions) < n:
        board = chess.Board()
        num_moves = random.randint(10, 50)

        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(random.choice(legal_moves))

        if not board.is_game_over():
            positions.append(board.copy())

    return positions


class RandomAgent:
    """Random move baseline agent"""

    def __call__(self, fen: str, legal_moves: List[str]) -> str:
        if not legal_moves:
            return "<uci_move>0000</uci_move>"
        move = random.choice(legal_moves)
        return f"<uci_move>{move}</uci_move>"


class GreedyCaptureAgent:
    """Agent that prefers captures (slightly better than random)"""

    def __call__(self, fen: str, legal_moves: List[str]) -> str:
        if not legal_moves:
            return "<uci_move>0000</uci_move>"

        board = chess.Board(fen)

        # Prefer captures
        captures = []
        non_captures = []

        for move_uci in legal_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if board.is_capture(move):
                    captures.append(move_uci)
                else:
                    non_captures.append(move_uci)
            except:
                non_captures.append(move_uci)

        # 80% chance to pick capture if available
        if captures and random.random() < 0.8:
            move = random.choice(captures)
        else:
            move = random.choice(legal_moves)

        return f"<uci_move>{move}</uci_move>"


class LLMAgent:
    """Agent using actual trained chess LLM"""

    def __init__(self, model_path: Path, base_model: str = "Qwen/Qwen2.5-3B"):
        if not HAVE_TORCH:
            raise ImportError("PyTorch not available for LLM agent")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading LLM on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map={"": self.device},
            trust_remote_code=True
        )

        if model_path.exists():
            print(f"Loading LoRA from {model_path}")
            self.model = PeftModel.from_pretrained(self.model, str(model_path))

        self.model.eval()

    def __call__(self, fen: str, legal_moves: List[str]) -> str:
        moves_str = ", ".join(legal_moves)

        messages = [
            {"role": "system", "content": "You are a chess grandmaster. Analyze positions and select the best move."},
            {"role": "user", "content": f"FEN: {fen}\nMoves: {moves_str}\nBest?"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text += "<uci_move>"

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = "<uci_move>" + self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response


def parse_uci_move(response: str) -> Optional[str]:
    """Extract UCI move from model response"""
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
    if match:
        return match.group(1)

    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)', response)
    if match:
        return match.group(1)

    return None


class RewardComputer:
    """Computes rewards for chess moves using Stockfish"""

    def __init__(self, depth: int = 12, max_workers: int = 2):
        self.depth = depth
        self.max_workers = max_workers
        # Use single engine to respect resource limits
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    def compute_reward(
        self,
        board: chess.Board,
        model_move_uci: str
    ) -> RewardResult:
        """
        Compute reward for a model's move.

        Reward = -centipawn_loss (higher is better)
        """
        fen = board.fen()
        mover_is_white = (board.turn == chess.WHITE)

        analysis_start = time.time()

        # Get best move and its evaluation
        root = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        best_move = root["pv"][0]

        # Score AFTER best move
        board.push(best_move)
        best_after = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        best_after_cp = score_to_cp(best_after["score"])
        board.pop()

        # Check legality of model move
        try:
            model_move = chess.Move.from_uci(model_move_uci)
            is_legal = model_move in board.legal_moves
        except ValueError:
            is_legal = False
            model_move = None

        analysis_time = time.time() - analysis_start

        if not is_legal:
            # Huge penalty for illegal move
            return RewardResult(
                fen=fen,
                model_move=model_move_uci,
                best_move=best_move.uci(),
                model_cp_after=-10000 if mover_is_white else 10000,
                best_cp_after=best_after_cp,
                cp_loss=10000,
                reward=-10000.0,  # Severe penalty
                is_legal=False,
                is_best=False,
                inference_time=0.0,
                analysis_time=analysis_time
            )

        # Score AFTER model move
        board.push(model_move)
        model_after = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        model_after_cp = score_to_cp(model_after["score"])
        board.pop()

        # CP loss from mover's perspective
        if mover_is_white:
            cp_loss = best_after_cp - model_after_cp
        else:
            cp_loss = model_after_cp - best_after_cp

        cp_loss = max(0, cp_loss)

        # Reward is negative CP loss (higher is better)
        # Scale to reasonable range for RL
        reward = -float(cp_loss)

        return RewardResult(
            fen=fen,
            model_move=model_move_uci,
            best_move=best_move.uci(),
            model_cp_after=model_after_cp,
            best_cp_after=best_after_cp,
            cp_loss=cp_loss,
            reward=reward,
            is_legal=True,
            is_best=(model_move == best_move),
            inference_time=0.0,
            analysis_time=analysis_time
        )

    def close(self):
        if hasattr(self, 'engine'):
            self.engine.quit()


def run_validation(
    agent: Callable,
    positions: List[chess.Board],
    depth: int = 12,
    verbose: bool = True
) -> Tuple[List[RewardResult], ValidationSummary]:
    """
    Run RL validation loop.

    For each position:
    1. Get model's move
    2. Compute reward via Stockfish
    3. Track metrics
    """
    results = []
    reward_computer = RewardComputer(depth=depth)

    total_start = time.time()

    for i, board in enumerate(positions):
        # Get legal moves
        legal_moves = [m.uci() for m in board.legal_moves]

        # Get model's move
        inference_start = time.time()
        response = agent(board.fen(), legal_moves)
        inference_time = time.time() - inference_start

        model_move = parse_uci_move(response)
        if model_move is None:
            model_move = "0000"  # Invalid

        # Compute reward
        result = reward_computer.compute_reward(board, model_move)
        result.inference_time = inference_time
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            legal_results = [r for r in results if r.is_legal]
            avg_reward = statistics.mean([r.reward for r in results])
            legal_rate = len(legal_results) / len(results)
            print(f"  Step {i+1}/{len(positions)}: "
                  f"avg_reward={avg_reward:.1f}, legal_rate={legal_rate:.1%}")

    total_time = time.time() - total_start
    reward_computer.close()

    # Compute summary statistics
    rewards = [r.reward for r in results]
    legal_results = [r for r in results if r.is_legal]

    # Check if reward signal is informative (has variance)
    reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    reward_signal_informative = reward_std > 50  # Meaningful variation

    summary = ValidationSummary(
        timestamp=datetime.now().isoformat(),
        num_positions=len(positions),
        stockfish_depth=depth,

        # Reward stats
        mean_reward=statistics.mean(rewards),
        std_reward=reward_std,
        min_reward=min(rewards),
        max_reward=max(rewards),
        median_reward=statistics.median(rewards),

        # Quality metrics
        legal_rate=len(legal_results) / len(results) if results else 0.0,
        best_move_rate=sum(1 for r in results if r.is_best) / len(results) if results else 0.0,
        mean_cp_loss=statistics.mean([r.cp_loss for r in legal_results]) if legal_results else 10000.0,

        # Timing
        total_time=total_time,
        avg_inference_time=statistics.mean([r.inference_time for r in results]) if results else 0.0,
        avg_analysis_time=statistics.mean([r.analysis_time for r in results]) if results else 0.0,

        # RL readiness assessment
        reward_signal_informative=reward_signal_informative,
        infrastructure_ready=True,
        recommendation=""
    )

    # Generate recommendation
    if summary.legal_rate < 0.95:
        summary.recommendation = "FAIL: Legal move rate too low. Fix output format before RL."
    elif not summary.reward_signal_informative:
        summary.recommendation = "CAUTION: Reward signal has low variance. May not guide learning well."
    elif summary.mean_cp_loss > 500:
        summary.recommendation = "PROCEED: Infrastructure ready. Model has room to improve via RL."
    else:
        summary.recommendation = "PROCEED: Infrastructure ready. Model already reasonably good."

    return results, summary


def create_reward_plot(results: List[RewardResult], output_path: Path):
    """Create visualization of rewards over time"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Rewards over steps
        rewards = [r.reward for r in results]
        steps = range(1, len(rewards) + 1)

        ax1 = axes[0, 0]
        ax1.plot(steps, rewards, 'b-', alpha=0.5, label='Individual')

        # Moving average
        window = min(20, len(rewards) // 5)
        if window > 1:
            moving_avg = []
            for i in range(len(rewards)):
                start = max(0, i - window)
                moving_avg.append(statistics.mean(rewards[start:i+1]))
            ax1.plot(steps, moving_avg, 'r-', linewidth=2, label=f'Moving avg (w={window})')

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward (negative CPL)')
        ax1.set_title('Rewards Over Validation Steps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Reward distribution
        ax2 = axes[0, 1]
        legal_rewards = [r.reward for r in results if r.is_legal]
        illegal_rewards = [r.reward for r in results if not r.is_legal]

        if legal_rewards:
            ax2.hist(legal_rewards, bins=30, alpha=0.7, label=f'Legal ({len(legal_rewards)})')
        if illegal_rewards:
            ax2.hist(illegal_rewards, bins=10, alpha=0.7, label=f'Illegal ({len(illegal_rewards)})')

        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Count')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: CP Loss distribution
        ax3 = axes[1, 0]
        cp_losses = [r.cp_loss for r in results if r.is_legal]
        if cp_losses:
            ax3.hist(cp_losses, bins=30, color='orange', alpha=0.7)
            ax3.axvline(statistics.mean(cp_losses), color='red', linestyle='--',
                       label=f'Mean: {statistics.mean(cp_losses):.0f}')
            ax3.axvline(statistics.median(cp_losses), color='green', linestyle='--',
                       label=f'Median: {statistics.median(cp_losses):.0f}')

        ax3.set_xlabel('Centipawn Loss')
        ax3.set_ylabel('Count')
        ax3.set_title('CP Loss Distribution (Legal Moves Only)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary metrics
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = [
            f"Validation Summary",
            f"=" * 40,
            f"Positions: {len(results)}",
            f"Legal Rate: {sum(1 for r in results if r.is_legal)/len(results):.1%}",
            f"Best Move Rate: {sum(1 for r in results if r.is_best)/len(results):.1%}",
            f"",
            f"Reward Statistics:",
            f"  Mean: {statistics.mean(rewards):.1f}",
            f"  Std: {statistics.stdev(rewards) if len(rewards) > 1 else 0:.1f}",
            f"  Min: {min(rewards):.1f}",
            f"  Max: {max(rewards):.1f}",
            f"",
            f"CP Loss (legal only):",
            f"  Mean: {statistics.mean(cp_losses) if cp_losses else 'N/A':.1f}" if cp_losses else "  Mean: N/A",
            f"  Median: {statistics.median(cp_losses) if cp_losses else 'N/A':.1f}" if cp_losses else "  Median: N/A",
        ]

        ax4.text(0.1, 0.95, '\n'.join(summary_text), transform=ax4.transAxes,
                fontfamily='monospace', fontsize=11, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Visualization saved to: {output_path}")

    except ImportError:
        print("  matplotlib not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description="Mini RL Validation for Chess LLM")
    parser.add_argument("--num-positions", type=int, default=100,
                       help="Number of positions to evaluate (default: 100)")
    parser.add_argument("--depth", type=int, default=12,
                       help="Stockfish analysis depth (default: 12 for speed)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to LoRA model (default: use random agent)")
    parser.add_argument("--agent", type=str, choices=["random", "greedy", "llm"],
                       default="random", help="Agent type to test")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 60)
    print("Mini RL Validation - Global Chess Challenge 2025")
    print("=" * 60)
    print()

    # Set seed
    random.seed(args.seed)

    # Load positions
    print("Loading positions...")
    training_data = DATA_DIR / "training_elite_30k.jsonl"

    if training_data.exists():
        print(f"  Using training data: {training_data}")
        positions = load_positions_from_training_data(
            training_data, n=args.num_positions, seed=args.seed
        )
        position_source = "training_elite_30k.jsonl"
    else:
        print("  WARNING: Training data not found, using random positions")
        positions = generate_random_positions(n=args.num_positions, seed=args.seed)
        position_source = "random_walk"

    print(f"  Loaded {len(positions)} positions from {position_source}")
    print()

    # Create agent
    print("Creating agent...")
    if args.agent == "llm" and args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_DIR / args.model_path
        agent = LLMAgent(model_path)
        agent_name = f"LLM ({model_path.name})"
    elif args.agent == "greedy":
        agent = GreedyCaptureAgent()
        agent_name = "GreedyCapture"
    else:
        agent = RandomAgent()
        agent_name = "Random"

    print(f"  Agent: {agent_name}")
    print()

    # Run validation
    print(f"Running validation ({args.num_positions} positions, depth={args.depth})...")
    print(f"  (Max 2 Stockfish workers for resource safety)")
    print()

    results, summary = run_validation(
        agent=agent,
        positions=positions,
        depth=args.depth,
        verbose=True
    )

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print()

    print(f"Agent: {agent_name}")
    print(f"Positions: {summary.num_positions}")
    print(f"Stockfish Depth: {summary.stockfish_depth}")
    print()

    print("Reward Statistics:")
    print(f"  Mean:   {summary.mean_reward:.1f}")
    print(f"  Std:    {summary.std_reward:.1f}")
    print(f"  Min:    {summary.min_reward:.1f}")
    print(f"  Max:    {summary.max_reward:.1f}")
    print(f"  Median: {summary.median_reward:.1f}")
    print()

    print("Quality Metrics:")
    print(f"  Legal Move Rate: {summary.legal_rate:.1%}")
    print(f"  Best Move Rate:  {summary.best_move_rate:.1%}")
    print(f"  Mean CP Loss:    {summary.mean_cp_loss:.1f}")
    print()

    print("Timing:")
    print(f"  Total Time:        {summary.total_time:.1f}s")
    print(f"  Avg Inference:     {summary.avg_inference_time:.3f}s")
    print(f"  Avg SF Analysis:   {summary.avg_analysis_time:.3f}s")
    print()

    print("RL Readiness Assessment:")
    print(f"  Reward Signal Informative: {summary.reward_signal_informative}")
    print(f"  Infrastructure Ready:      {summary.infrastructure_ready}")
    print()
    print(f"  RECOMMENDATION: {summary.recommendation}")
    print()

    # Save results
    output_data = {
        "summary": asdict(summary),
        "agent": agent_name,
        "position_source": position_source,
        "config": {
            "num_positions": args.num_positions,
            "depth": args.depth,
            "seed": args.seed,
            "model_path": args.model_path
        },
        "detailed_results": [asdict(r) for r in results]
    }

    output_path = RESULTS_DIR / "mini_rl_validation.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Create visualization
    vis_path = RESULTS_DIR / "mini_rl_validation.png"
    create_reward_plot(results, vis_path)

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    # Exit code based on validation
    if "FAIL" in summary.recommendation:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
