#!/usr/bin/env python3
"""
Re-ranking script using trained pairwise ranking model.

Given multiple candidate packings for the same n, uses pairwise comparisons
to rank them and select the best one.

Usage:
    python rerank_with_model.py --model ranking_model.pt --candidates candidates.jsonl

Candidate format (JSONL):
    {"n": 10, "side_length": 2.5, "positions": [[x,y,r], ...]}
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple

from pairwise_model import PairwiseRankingNet


MAX_N = 30  # Model was trained with max_n=30


def load_model(model_path: str, device: torch.device) -> PairwiseRankingNet:
    """Load trained pairwise ranking model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    max_n = checkpoint.get('max_n', MAX_N)
    model = PairwiseRankingNet(max_n=max_n)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
    print(f"  Max N: {max_n}")

    return model, max_n


def extract_features(packing: Dict[str, Any], max_n: int = MAX_N) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert packing to model input features.

    Args:
        packing: {"n": int, "side_length": float, "positions": [[x, y, rot], ...]}
        max_n: Maximum number of trees the model supports

    Returns:
        features: [max_n * 3 + 1] tensor
        target_n: [1] tensor (normalized n)
    """
    n = packing['n']
    positions = packing['positions']
    side_length = packing['side_length']

    # Normalize n
    n_normalized = n / max_n
    target_n = torch.tensor([n_normalized], dtype=torch.float32)

    # Build feature vector: n_trees (normalized), then up to max_n trees with (x, y, rot)
    # Order matches generate_training_data.rs: x/10, y/10, angle_deg/360
    features = [n_normalized]

    for i in range(max_n):
        if i < len(positions):
            x, y, rot = positions[i]
            # Normalize to match training data format
            x_norm = x / 10.0
            y_norm = y / 10.0
            rot_norm = rot / 360.0
            features.extend([x_norm, y_norm, rot_norm])
        else:
            features.extend([0.0, 0.0, 0.0])

    return torch.tensor(features, dtype=torch.float32), target_n


def compare_pair(
    model: PairwiseRankingNet,
    packing_a: Dict[str, Any],
    packing_b: Dict[str, Any],
    max_n: int,
    device: torch.device
) -> float:
    """
    Compare two packings using the model.

    Returns: P(A is better than B), where "better" means smaller side_length
    """
    features_a, target_n = extract_features(packing_a, max_n)
    features_b, _ = extract_features(packing_b, max_n)

    # Add batch dimension
    features_a = features_a.unsqueeze(0).to(device)
    features_b = features_b.unsqueeze(0).to(device)
    target_n = target_n.unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(features_a, features_b, target_n)

    return prob.item()


def rank_candidates_round_robin(
    model: PairwiseRankingNet,
    candidates: List[Dict[str, Any]],
    max_n: int,
    device: torch.device
) -> List[Tuple[int, float]]:
    """
    Rank candidates using round-robin pairwise comparisons.

    Returns: List of (candidate_index, win_rate) sorted by win_rate descending
    """
    n_candidates = len(candidates)
    if n_candidates == 1:
        return [(0, 1.0)]

    wins = [0.0] * n_candidates

    # Compare all pairs
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            p_i_better = compare_pair(model, candidates[i], candidates[j], max_n, device)
            wins[i] += p_i_better
            wins[j] += (1 - p_i_better)

    # Calculate win rates (normalized by number of comparisons)
    n_comparisons = n_candidates - 1
    win_rates = [(i, wins[i] / n_comparisons) for i in range(n_candidates)]

    # Sort by win rate descending
    win_rates.sort(key=lambda x: x[1], reverse=True)

    return win_rates


def rank_candidates_batch(
    model: PairwiseRankingNet,
    candidates: List[Dict[str, Any]],
    max_n: int,
    device: torch.device
) -> List[Tuple[int, float]]:
    """
    Rank candidates using batched pairwise comparisons (faster for many candidates).
    """
    n_candidates = len(candidates)
    if n_candidates == 1:
        return [(0, 1.0)]

    # Pre-compute all features
    features_list = []
    target_n_list = []
    for c in candidates:
        f, t = extract_features(c, max_n)
        features_list.append(f)
        target_n_list.append(t)

    features = torch.stack(features_list).to(device)
    target_n = target_n_list[0].unsqueeze(0).to(device)  # Same for all

    # Build all pairs
    pairs_i = []
    pairs_j = []
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            pairs_i.append(i)
            pairs_j.append(j)

    if len(pairs_i) == 0:
        return [(0, 1.0)]

    # Batch comparison
    features_a = features[pairs_i]
    features_b = features[pairs_j]
    target_n_batch = target_n.expand(len(pairs_i), -1)

    with torch.no_grad():
        probs = model(features_a, features_b, target_n_batch)

    # Count wins
    wins = [0.0] * n_candidates
    for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        p_i_better = probs[idx].item()
        wins[i] += p_i_better
        wins[j] += (1 - p_i_better)

    # Normalize
    n_comparisons = n_candidates - 1
    win_rates = [(i, wins[i] / n_comparisons) for i in range(n_candidates)]
    win_rates.sort(key=lambda x: x[1], reverse=True)

    return win_rates


def select_best(
    model: PairwiseRankingNet,
    candidates: List[Dict[str, Any]],
    max_n: int,
    device: torch.device
) -> Tuple[int, Dict[str, Any]]:
    """
    Select the best candidate according to the model.

    Returns: (best_index, best_candidate)
    """
    rankings = rank_candidates_batch(model, candidates, max_n, device)
    best_idx = rankings[0][0]
    return best_idx, candidates[best_idx]


def main():
    parser = argparse.ArgumentParser(description='Re-rank packings using ML model')
    parser.add_argument('--model', type=str, default='ranking_model.pt',
                        help='Path to trained model')
    parser.add_argument('--candidates', type=str, required=True,
                        help='Path to candidates JSONL file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for ranked results (default: stdout)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed comparison info')
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model, max_n = load_model(args.model, device)

    # Load candidates grouped by n
    candidates_by_n: Dict[int, List[Dict[str, Any]]] = {}
    with open(args.candidates, 'r') as f:
        for line in f:
            packing = json.loads(line)
            n = packing['n']
            if n not in candidates_by_n:
                candidates_by_n[n] = []
            candidates_by_n[n].append(packing)

    print(f"Loaded candidates for {len(candidates_by_n)} different n values")

    results = []
    total_ml_wins = 0
    total_pure_wins = 0
    total_ties = 0

    for n in sorted(candidates_by_n.keys()):
        candidates = candidates_by_n[n]
        if len(candidates) == 1:
            results.append(candidates[0])
            continue

        # Get ML ranking
        rankings = rank_candidates_batch(model, candidates, max_n, device)
        ml_best_idx = rankings[0][0]
        ml_best = candidates[ml_best_idx]

        # Get pure best-of-N (smallest side_length)
        pure_best_idx = min(range(len(candidates)),
                           key=lambda i: candidates[i]['side_length'])
        pure_best = candidates[pure_best_idx]

        # Compare
        if ml_best['side_length'] < pure_best['side_length']:
            total_ml_wins += 1
            winner = "ML"
        elif ml_best['side_length'] > pure_best['side_length']:
            total_pure_wins += 1
            winner = "Pure"
        else:
            total_ties += 1
            winner = "Tie"

        if args.verbose:
            print(f"n={n}: ML selected idx={ml_best_idx} (side={ml_best['side_length']:.4f}), "
                  f"Pure selected idx={pure_best_idx} (side={pure_best['side_length']:.4f}) -> {winner}")

        results.append(ml_best)

    # Summary
    total = total_ml_wins + total_pure_wins + total_ties
    print(f"\nSummary:")
    print(f"  ML wins: {total_ml_wins}/{total} ({100*total_ml_wins/total:.1f}%)")
    print(f"  Pure wins: {total_pure_wins}/{total} ({100*total_pure_wins/total:.1f}%)")
    print(f"  Ties: {total_ties}/{total} ({100*total_ties/total:.1f}%)")

    # Calculate total scores
    ml_total = sum(r['side_length'] for r in results)

    # Recalculate pure total
    pure_results = []
    for n in sorted(candidates_by_n.keys()):
        candidates = candidates_by_n[n]
        pure_best = min(candidates, key=lambda c: c['side_length'])
        pure_results.append(pure_best)
    pure_total = sum(r['side_length'] for r in pure_results)

    print(f"\nTotal scores:")
    print(f"  ML selection: {ml_total:.4f}")
    print(f"  Pure best-of-N: {pure_total:.4f}")
    print(f"  Difference: {ml_total - pure_total:.4f} ({100*(ml_total/pure_total - 1):.2f}%)")

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"\nResults written to {args.output}")


if __name__ == '__main__':
    main()
