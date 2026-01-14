#!/usr/bin/env python3
"""
Fitness evaluator for sales territory optimization.

Evaluates a solution's assign_territories() function and returns
a fitness score based on coverage, balance, and compactness.

Usage:
    python evaluate.py solution.py --json
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

from data import get_accounts


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two lat/lon points."""
    R = 3959  # Earth radius in miles

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def load_solution(solution_path: str):
    """Dynamically load a solution module."""
    path = Path(solution_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def calculate_coverage_score(accounts: list[dict], assignments: dict[int, list[int]]) -> float:
    """
    Calculate coverage score: % of total revenue assigned.
    Perfect score = 1.0 (all accounts assigned).
    """
    total_revenue = sum(a["revenue"] for a in accounts)
    assigned_ids = set()
    for account_ids in assignments.values():
        assigned_ids.update(account_ids)

    assigned_revenue = sum(a["revenue"] for a in accounts if a["id"] in assigned_ids)
    return assigned_revenue / total_revenue if total_revenue > 0 else 0.0


def calculate_balance_score(accounts: list[dict], assignments: dict[int, list[int]]) -> float:
    """
    Calculate workload balance score using coefficient of variation.
    Perfect balance = 1.0, worse balance = lower score.

    Uses revenue as workload metric.
    """
    account_map = {a["id"]: a for a in accounts}
    rep_revenues = []

    for rep_id, account_ids in assignments.items():
        rep_revenue = sum(account_map[aid]["revenue"] for aid in account_ids if aid in account_map)
        rep_revenues.append(rep_revenue)

    if not rep_revenues or all(r == 0 for r in rep_revenues):
        return 0.0

    mean_rev = sum(rep_revenues) / len(rep_revenues)
    if mean_rev == 0:
        return 0.0

    # Coefficient of variation (std / mean)
    variance = sum((r - mean_rev) ** 2 for r in rep_revenues) / len(rep_revenues)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_rev

    # Convert to score (lower CV = better balance = higher score)
    # CV of 0 = perfect balance = 1.0
    # CV of 1 = very unbalanced = 0.0
    return max(0.0, 1.0 - cv)


def calculate_compactness_score(accounts: list[dict], assignments: dict[int, list[int]]) -> float:
    """
    Calculate territory compactness score.
    Measures how geographically tight each territory is.

    Uses average pairwise distance within territories.
    Lower distance = more compact = higher score.
    """
    account_map = {a["id"]: a for a in accounts}
    territory_compactness = []

    for rep_id, account_ids in assignments.items():
        if len(account_ids) < 2:
            territory_compactness.append(1.0)  # Single account = perfectly compact
            continue

        # Calculate average pairwise distance
        rep_accounts = [account_map[aid] for aid in account_ids if aid in account_map]
        total_dist = 0
        pairs = 0

        for i, a1 in enumerate(rep_accounts):
            for a2 in rep_accounts[i + 1:]:
                dist = haversine_distance(a1["lat"], a1["lon"], a2["lat"], a2["lon"])
                total_dist += dist
                pairs += 1

        avg_dist = total_dist / pairs if pairs > 0 else 0

        # Convert distance to compactness score
        # 0 miles avg = 1.0, 30 miles avg = 0.0
        compactness = max(0.0, 1.0 - avg_dist / 30.0)
        territory_compactness.append(compactness)

    return sum(territory_compactness) / len(territory_compactness) if territory_compactness else 0.0


def validate_assignments(accounts: list[dict], assignments: dict[int, list[int]], num_reps: int) -> tuple[bool, list[str]]:
    """
    Validate that assignments are valid.

    Rules:
    - Each account assigned to exactly one rep (or none)
    - No duplicate assignments
    - Account IDs must exist
    """
    errors = []
    account_ids = {a["id"] for a in accounts}
    seen_accounts = set()

    for rep_id, rep_account_ids in assignments.items():
        for aid in rep_account_ids:
            if aid not in account_ids:
                errors.append(f"Invalid account ID {aid} assigned to rep {rep_id}")
            if aid in seen_accounts:
                errors.append(f"Account {aid} assigned to multiple reps")
            seen_accounts.add(aid)

    return len(errors) == 0, errors


def evaluate_solution(solution_path: str, num_reps: int = 4) -> dict:
    """
    Evaluate a solution file and return fitness metrics.

    Args:
        solution_path: Path to Python file with assign_territories() function
        num_reps: Number of sales reps

    Returns:
        Dict with valid, fitness, and component scores
    """
    accounts = get_accounts()

    try:
        module = load_solution(solution_path)

        if not hasattr(module, "assign_territories"):
            return {
                "valid": False,
                "error": "Solution must have assign_territories(accounts, num_reps) function",
                "fitness": 0.0,
            }

        assignments = module.assign_territories(accounts, num_reps)

        # Validate
        valid, errors = validate_assignments(accounts, assignments, num_reps)
        if not valid:
            return {
                "valid": False,
                "errors": errors,
                "fitness": 0.0,
            }

        # Calculate component scores
        coverage = calculate_coverage_score(accounts, assignments)
        balance = calculate_balance_score(accounts, assignments)
        compactness = calculate_compactness_score(accounts, assignments)

        # Combined fitness (weighted)
        fitness = (coverage * 0.4) + (balance * 0.3) + (compactness * 0.3)

        # Build detailed results
        account_map = {a["id"]: a for a in accounts}
        rep_details = {}
        for rep_id, account_ids in assignments.items():
            rep_accounts = [account_map[aid] for aid in account_ids if aid in account_map]
            rep_details[f"rep_{rep_id}"] = {
                "account_count": len(account_ids),
                "total_revenue": sum(a["revenue"] for a in rep_accounts),
                "industries": list(set(a["industry"] for a in rep_accounts)),
            }

        return {
            "valid": True,
            "fitness": round(fitness, 4),
            "coverage_score": round(coverage, 4),
            "balance_score": round(balance, 4),
            "compactness_score": round(compactness, 4),
            "rep_details": rep_details,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "fitness": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate territory assignment solution")
    parser.add_argument("solution", help="Path to solution Python file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--reps", type=int, default=4, help="Number of sales reps")

    args = parser.parse_args()

    result = evaluate_solution(args.solution, args.reps)

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Valid: {result['valid']}")
        print(f"Fitness: {result.get('fitness', 0):.4f}")
        if result["valid"]:
            print(f"  Coverage:   {result['coverage_score']:.4f}")
            print(f"  Balance:    {result['balance_score']:.4f}")
            print(f"  Compactness: {result['compactness_score']:.4f}")
            print("\nRep Details:")
            for rep, details in result.get("rep_details", {}).items():
                print(f"  {rep}: {details['account_count']} accounts, ${details['total_revenue']:,}")
        else:
            print(f"Error: {result.get('error', result.get('errors', 'Unknown'))}")


if __name__ == "__main__":
    main()
