#!/usr/bin/env python3
"""
Gen109 Runner - Final Submission Generator

This script orchestrates:
1. Rust Gen109 algorithm (best-of-N for each n)
2. ILP solver for small n (n <= 10)
3. Best selection and validation
4. CSV submission generation
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Tree vertices for validation
TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]


def rotate_point(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def get_tree_vertices(x: float, y: float, angle_deg: float) -> List[Tuple[float, float]]:
    angle_rad = angle_deg * math.pi / 180.0
    return [(rotate_point(vx, vy, angle_rad)[0] + x,
             rotate_point(vx, vy, angle_rad)[1] + y)
            for vx, vy in TREE_VERTICES]


def compute_side_length(trees: List[Tuple[float, float, float]]) -> float:
    """Compute side length of bounding square for trees."""
    if not trees:
        return 0.0

    all_vertices = []
    for x, y, a in trees:
        all_vertices.extend(get_tree_vertices(x, y, a))

    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return max(width, height)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    d1, d2 = ccw(A, B, C), ccw(A, B, D)
    d3, d4 = ccw(C, D, A), ccw(C, D, B)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygons_overlap(poly1, poly2):
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(poly1[i], poly1[(i+1) % n1], poly2[j], poly2[(j+1) % n2]):
                return True
    for v in poly1:
        if point_in_polygon(v, poly2):
            return True
    for v in poly2:
        if point_in_polygon(v, poly1):
            return True
    return False


def validate_packing(trees: List[Tuple[float, float, float]]) -> Tuple[bool, float]:
    """Validate no overlaps and return side length."""
    n = len(trees)
    for i in range(n):
        for j in range(i + 1, n):
            v1 = get_tree_vertices(*trees[i])
            v2 = get_tree_vertices(*trees[j])
            if polygons_overlap(v1, v2):
                return False, float('inf')
    return True, compute_side_length(trees)


def run_rust_best_of_n(n_max: int, num_runs: int = 20) -> Dict[int, Tuple[float, List[Tuple[float, float, float]]]]:
    """Run Rust best-of-N and return best packings for each n."""
    script_dir = Path(__file__).parent.parent
    rust_bin = script_dir / "rust" / "target" / "release" / "export_all"

    if not rust_bin.exists():
        # Build if needed
        print("Building Rust...")
        subprocess.run(["cargo", "build", "--release", "--bin", "export_all"],
                       cwd=script_dir / "rust", check=True)

    print(f"Running Rust best-of-{num_runs} for n=1..{n_max}...")
    start = time.time()

    # Run export_all which outputs JSON to stdout
    result = subprocess.run(
        [str(rust_bin), str(n_max), str(num_runs)],
        capture_output=True, text=True, cwd=script_dir / "rust"
    )

    if result.returncode != 0:
        print(f"Rust failed: {result.stderr}")
        return {}

    # Parse JSON output
    try:
        data = json.loads(result.stdout)
        packings = {}
        for n_data in data['packings']:
            n = n_data['n']
            trees = [(t['x'], t['y'], t['angle']) for t in n_data['trees']]
            side = n_data['side']
            packings[n] = (side, trees)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s, Rust score: {sum(s*s/n for n,(s,_) in packings.items()):.4f}")
        return packings
    except json.JSONDecodeError as e:
        print(f"Failed to parse Rust output: {e}")
        print(f"First 500 chars: {result.stdout[:500]}")
        return {}


def run_ilp_for_small_n(max_n: int = 10, timeout: float = 60.0) -> Dict[int, Tuple[float, List[Tuple[float, float, float]]]]:
    """Run ILP solver for small n."""
    script_dir = Path(__file__).parent.parent
    venv_python = script_dir / "venv" / "bin" / "python"
    ilp_script = script_dir / "python" / "ilp_solver.py"

    if not venv_python.exists():
        print("venv not found, skipping ILP")
        return {}

    packings = {}
    for n in range(1, max_n + 1):
        print(f"  ILP solving n={n}...")

        result = subprocess.run(
            [str(venv_python), str(ilp_script), str(n),
             "--timeout", str(timeout), "--refine",
             "--output", f"/tmp/ilp_n{n}.json"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            try:
                with open(f"/tmp/ilp_n{n}.json") as f:
                    data = json.load(f)
                trees = [(t['x'], t['y'], t['angle']) for t in data['trees']]
                valid, side = validate_packing(trees)
                if valid:
                    packings[n] = (side, trees)
                    print(f"    Found: side={side:.4f}")
            except:
                pass

    return packings


def select_best(rust_packings: Dict, ilp_packings: Dict) -> Dict[int, Tuple[float, List]]:
    """Select best solution for each n from all sources."""
    best = {}

    all_n = set(rust_packings.keys()) | set(ilp_packings.keys())

    for n in sorted(all_n):
        candidates = []

        if n in rust_packings:
            candidates.append(('rust', rust_packings[n]))
        if n in ilp_packings:
            candidates.append(('ilp', ilp_packings[n]))

        if candidates:
            best_source, (best_side, best_trees) = min(candidates, key=lambda x: x[1][0])
            best[n] = (best_side, best_trees)

            if len(candidates) > 1:
                rust_side = rust_packings.get(n, (float('inf'), []))[0]
                ilp_side = ilp_packings.get(n, (float('inf'), []))[0]
                print(f"  n={n}: rust={rust_side:.4f}, ilp={ilp_side:.4f} -> using {best_source}")

    return best


def generate_submission(packings: Dict[int, Tuple[float, List]], output_file: str):
    """Generate CSV submission file."""
    with open(output_file, 'w') as f:
        f.write("row_id,x,y,angle\n")

        row_id = 0
        for n in range(1, max(packings.keys()) + 1):
            if n not in packings:
                print(f"WARNING: Missing n={n}")
                continue

            _, trees = packings[n]
            if len(trees) != n:
                print(f"WARNING: n={n} has {len(trees)} trees")
                continue

            for x, y, a in trees:
                f.write(f"{row_id},{x},{y},{a}\n")
                row_id += 1

    print(f"Wrote {row_id} rows to {output_file}")


def compute_score(packings: Dict[int, Tuple[float, List]]) -> float:
    """Compute total score."""
    total = 0.0
    for n, (side, _) in packings.items():
        total += side * side / n
    return total


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Gen109 Runner')
    parser.add_argument('--max-n', type=int, default=200, help='Max n')
    parser.add_argument('--rust-runs', type=int, default=20, help='Best-of-N for Rust')
    parser.add_argument('--ilp-max-n', type=int, default=10, help='Max n for ILP')
    parser.add_argument('--ilp-timeout', type=float, default=60, help='ILP timeout per n')
    parser.add_argument('--output', type=str, default='submission_gen109.csv', help='Output file')
    parser.add_argument('--skip-ilp', action='store_true', help='Skip ILP solver')
    parser.add_argument('--skip-rust', action='store_true', help='Skip Rust (use cached)')

    args = parser.parse_args()

    print("=" * 60)
    print("Gen109 Runner - Final Submission Generator")
    print("=" * 60)

    # Run Rust
    rust_packings = {}
    if not args.skip_rust:
        rust_packings = run_rust_best_of_n(args.max_n, args.rust_runs)

    # Run ILP
    ilp_packings = {}
    if not args.skip_ilp and args.ilp_max_n > 0:
        print(f"\nRunning ILP for n=1..{args.ilp_max_n}...")
        ilp_packings = run_ilp_for_small_n(args.ilp_max_n, args.ilp_timeout)

    # Select best
    print("\nSelecting best solutions...")
    best_packings = select_best(rust_packings, ilp_packings)

    # Compute score
    score = compute_score(best_packings)
    print(f"\nTotal score: {score:.4f}")

    # Generate submission
    script_dir = Path(__file__).parent.parent
    output_path = script_dir / args.output
    generate_submission(best_packings, str(output_path))

    print("\nDone!")
    print(f"Score: {score:.4f}")

    # Show breakdown
    print("\nScore breakdown (top contributors):")
    contributions = [(n, side * side / n) for n, (side, _) in best_packings.items()]
    contributions.sort(key=lambda x: -x[1])
    for n, contrib in contributions[:10]:
        side = best_packings[n][0]
        print(f"  n={n:3d}: side={side:.4f}, contrib={contrib:.4f}")


if __name__ == '__main__':
    main()
