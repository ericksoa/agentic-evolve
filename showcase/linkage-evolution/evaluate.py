#!/usr/bin/env python3
"""
Linkage Path Evaluator

Evaluates how well a 4-bar linkage traces a target shape.

Fitness is based on:
1. Path accuracy (primary): How close the traced path is to the target
2. Coverage (secondary): What fraction of the target is traced
3. Validity bonus: Whether the linkage can complete its motion

Higher fitness = better linkage.
"""

import json
import sys
import math
from pathlib import Path
from typing import List, Tuple, Optional

from linkage import Linkage4Bar, load_linkage
from targets import get_target


def hausdorff_distance(path1: List[Tuple[float, float]], path2: List[Tuple[float, float]]) -> float:
    """
    Compute the Hausdorff distance between two paths.
    This measures the maximum distance from any point on one path to the closest point on the other.
    """
    if not path1 or not path2:
        return float('inf')

    def point_to_path_distance(point: Tuple[float, float], path: List[Tuple[float, float]]) -> float:
        px, py = point
        min_dist = float('inf')
        for qx, qy in path:
            dist = math.sqrt((px - qx)**2 + (py - qy)**2)
            min_dist = min(min_dist, dist)
        return min_dist

    # Max distance from path1 to path2
    max_dist_1_to_2 = max(point_to_path_distance(p, path2) for p in path1)

    # Max distance from path2 to path1
    max_dist_2_to_1 = max(point_to_path_distance(p, path1) for p in path2)

    return max(max_dist_1_to_2, max_dist_2_to_1)


def average_distance(traced: List[Tuple[float, float]], target: List[Tuple[float, float]]) -> float:
    """
    Compute average distance from traced path to target.
    For each traced point, find distance to closest target point.
    """
    if not traced or not target:
        return float('inf')

    total_dist = 0.0
    for tx, ty in traced:
        min_dist = float('inf')
        for gx, gy in target:
            dist = math.sqrt((tx - gx)**2 + (ty - gy)**2)
            min_dist = min(min_dist, dist)
        total_dist += min_dist

    return total_dist / len(traced)


def coverage_score(traced: List[Tuple[float, float]], target: List[Tuple[float, float]], threshold: float = 0.2) -> float:
    """
    Compute what fraction of target points are "covered" by traced path.
    A target point is covered if there's a traced point within threshold distance.
    """
    if not traced or not target:
        return 0.0

    covered = 0
    for gx, gy in target:
        for tx, ty in traced:
            dist = math.sqrt((tx - gx)**2 + (ty - gy)**2)
            if dist < threshold:
                covered += 1
                break

    return covered / len(target)


def normalize_path(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Normalize path to have centroid at origin and unit scale."""
    if not path:
        return path

    # Compute centroid
    cx = sum(p[0] for p in path) / len(path)
    cy = sum(p[1] for p in path) / len(path)

    # Compute scale (max distance from centroid)
    max_dist = max(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in path)
    if max_dist < 0.001:
        max_dist = 1.0

    # Normalize
    return [((p[0]-cx)/max_dist, (p[1]-cy)/max_dist) for p in path]


def evaluate_linkage(linkage: Linkage4Bar, target_name: str = "figure8",
                     num_samples: int = 200) -> dict:
    """
    Evaluate a linkage against a target shape.

    Returns dict with:
    - valid: bool
    - fitness: float (higher is better)
    - avg_distance: float (lower is better)
    - coverage: float (0-1, higher is better)
    - traced_points: int
    - error: str (if invalid)
    """
    # Check linkage validity
    is_valid, msg = linkage.is_valid()
    if not is_valid:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": f"Invalid linkage: {msg}"
        }

    # Get target path
    try:
        target = get_target(target_name, num_points=num_samples)
    except Exception as e:
        return {
            "valid": False,
            "fitness": 0.0,
            "error": f"Invalid target: {e}"
        }

    # Trace the linkage path
    traced = linkage.trace_path(num_samples=num_samples)

    if len(traced) < 10:
        return {
            "valid": False,
            "fitness": 0.0,
            "traced_points": len(traced),
            "error": "Linkage could not complete enough of its motion"
        }

    # Normalize both paths for fair comparison
    traced_norm = normalize_path(traced)
    target_norm = normalize_path(target)

    # Compute metrics
    avg_dist = average_distance(traced_norm, target_norm)
    coverage = coverage_score(traced_norm, target_norm, threshold=0.15)
    hausdorff = hausdorff_distance(traced_norm, target_norm)

    # Fitness: combine metrics (higher is better)
    # - Base fitness from inverse average distance
    # - Bonus for coverage
    # - Penalty for high Hausdorff (worst-case deviation)

    if avg_dist < 0.001:
        avg_dist = 0.001  # Prevent division by zero

    # Fitness formula:
    # - 100 / avg_dist gives high scores for low distances
    # - * coverage rewards full coverage
    # - / (1 + hausdorff) penalizes worst-case deviations
    fitness = (100.0 / (1.0 + avg_dist * 10)) * (0.5 + 0.5 * coverage) / (1.0 + hausdorff * 0.5)

    # Bonus for completing full rotation
    if linkage.can_complete_rotation():
        fitness *= 1.1

    return {
        "valid": True,
        "fitness": round(fitness, 4),
        "avg_distance": round(avg_dist, 4),
        "coverage": round(coverage, 4),
        "hausdorff": round(hausdorff, 4),
        "traced_points": len(traced),
        "can_rotate": linkage.can_complete_rotation(),
        "target": target_name
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <linkage.py> [--json] [--target=<name>]", file=sys.stderr)
        print("Available targets: heart, figure8, line, circle, ellipse, star, letter_d", file=sys.stderr)
        sys.exit(1)

    linkage_path = sys.argv[1]
    json_output = "--json" in sys.argv

    # Parse target name
    target_name = "figure8"  # default
    for arg in sys.argv:
        if arg.startswith("--target="):
            target_name = arg.split("=")[1]

    try:
        linkage = load_linkage(linkage_path)
    except Exception as e:
        result = {
            "valid": False,
            "fitness": 0.0,
            "error": f"Failed to load linkage: {e}"
        }
        if json_output:
            print(json.dumps(result))
        else:
            print(f"ERROR: {result['error']}")
        sys.exit(1)

    result = evaluate_linkage(linkage, target_name=target_name)

    if json_output:
        print(json.dumps(result))
    else:
        if result["valid"]:
            print(f"Linkage evaluation for target '{result.get('target', target_name)}':")
            print(f"  Fitness: {result['fitness']:.4f}")
            print(f"  Avg Distance: {result['avg_distance']:.4f}")
            print(f"  Coverage: {result['coverage']*100:.1f}%")
            print(f"  Hausdorff: {result['hausdorff']:.4f}")
            print(f"  Traced Points: {result['traced_points']}")
            print(f"  Full Rotation: {result['can_rotate']}")
        else:
            print(f"INVALID: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
