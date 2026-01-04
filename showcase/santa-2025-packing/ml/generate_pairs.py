"""
Generate pairwise training data from existing training data.

For each N value, creates pairs (packing_A, packing_B, label)
where label = 1 if A is better (smaller final_side).
"""

import json
import random
from collections import defaultdict
from pathlib import Path
import argparse


def load_data(jsonl_path: str) -> list:
    """Load training data from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def generate_pairs(
    samples: list,
    pairs_per_n: int = 100,
    margin: float = 0.01
) -> list:
    """
    Generate pairwise training data.

    Args:
        samples: List of training samples
        pairs_per_n: Number of pairs to generate per N value
        margin: Minimum difference in final_side to create a pair

    Returns:
        List of (features_a, features_b, target_n, label) tuples
    """
    # Group samples by n and num_placed (only use final packings)
    by_n = defaultdict(list)
    for sample in samples:
        n = sample['n']
        # Use final packings (num_placed == n)
        if sample['num_placed'] == n:
            by_n[n].append(sample)

    print(f"Found samples for {len(by_n)} different N values")
    for n in sorted(by_n.keys())[:5]:
        print(f"  N={n}: {len(by_n[n])} final packings")

    pairs = []

    for n, n_samples in by_n.items():
        if len(n_samples) < 2:
            continue

        # Sort by final_side for easier pair generation
        n_samples.sort(key=lambda x: x['final_side'])

        # Generate pairs
        num_pairs = min(pairs_per_n, len(n_samples) * (len(n_samples) - 1) // 2)

        for _ in range(num_pairs):
            # Pick two different samples
            i, j = random.sample(range(len(n_samples)), 2)
            a, b = n_samples[i], n_samples[j]

            # Skip if too similar
            if abs(a['final_side'] - b['final_side']) < margin:
                continue

            # Label: 1 if A is better (smaller side)
            label = 1.0 if a['final_side'] < b['final_side'] else 0.0

            pairs.append({
                'features_a': a['features'],
                'features_b': b['features'],
                'target_n': a['target_n'],
                'label': label,
                'n': n,
                'side_a': a['final_side'],
                'side_b': b['final_side'],
            })

    # Shuffle pairs
    random.shuffle(pairs)

    print(f"\nGenerated {len(pairs)} pairs")
    return pairs


def generate_hard_pairs(
    samples: list,
    pairs_per_n: int = 100,
    percentile_range: tuple = (0.1, 0.9)
) -> list:
    """
    Generate harder pairwise training data by comparing good vs good.

    Instead of comparing worst vs best (easy), compare top 10% vs top 30%.
    This teaches finer distinctions.
    """
    by_n = defaultdict(list)
    for sample in samples:
        n = sample['n']
        if sample['num_placed'] == n:
            by_n[n].append(sample)

    pairs = []

    for n, n_samples in by_n.items():
        if len(n_samples) < 10:
            continue

        # Sort by final_side
        n_samples.sort(key=lambda x: x['final_side'])

        # Define percentile boundaries
        top_cutoff = int(len(n_samples) * percentile_range[0])
        mid_cutoff = int(len(n_samples) * percentile_range[1])

        top_samples = n_samples[:max(1, top_cutoff)]
        mid_samples = n_samples[top_cutoff:mid_cutoff]

        if not top_samples or not mid_samples:
            continue

        for _ in range(pairs_per_n):
            # Pick one from top, one from middle
            a = random.choice(top_samples)
            b = random.choice(mid_samples)

            # A is always better (smaller side)
            pairs.append({
                'features_a': a['features'],
                'features_b': b['features'],
                'target_n': a['target_n'],
                'label': 1.0,
                'n': n,
                'side_a': a['final_side'],
                'side_b': b['final_side'],
            })

            # Also add reverse
            pairs.append({
                'features_a': b['features'],
                'features_b': a['features'],
                'target_n': a['target_n'],
                'label': 0.0,
                'n': n,
                'side_a': b['final_side'],
                'side_b': a['final_side'],
            })

    random.shuffle(pairs)
    print(f"\nGenerated {len(pairs)} hard pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='training_data.jsonl',
                        help='Input training data JSONL')
    parser.add_argument('--output', type=str, default='pairwise_data.jsonl',
                        help='Output pairwise data JSONL')
    parser.add_argument('--pairs-per-n', type=int, default=200,
                        help='Number of pairs per N value')
    parser.add_argument('--hard', action='store_true',
                        help='Generate hard pairs (top vs middle)')
    parser.add_argument('--margin', type=float, default=0.001,
                        help='Minimum side length difference')
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    samples = load_data(args.input)
    print(f"Loaded {len(samples)} samples")

    if args.hard:
        pairs = generate_hard_pairs(samples, pairs_per_n=args.pairs_per_n)
    else:
        pairs = generate_pairs(samples, pairs_per_n=args.pairs_per_n, margin=args.margin)

    # Write output
    print(f"Writing to {args.output}...")
    with open(args.output, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print("Done!")

    # Print statistics
    labels = [p['label'] for p in pairs]
    print(f"\nStatistics:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Label=1 (A better): {sum(labels)} ({sum(labels)/len(pairs)*100:.1f}%)")
    print(f"  Label=0 (B better): {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(pairs)*100:.1f}%)")


if __name__ == '__main__':
    main()
