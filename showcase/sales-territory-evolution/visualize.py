#!/usr/bin/env python3
"""
Visualize sales territory assignments on a map.

Creates a scatter plot showing account locations colored by assigned rep,
with territory boundaries approximated by convex hulls.

Usage:
    python visualize.py solution.py [--output territories.png]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from data import get_accounts


# Color palette for territories
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
REP_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]


def load_solution(solution_path: str):
    """Dynamically load a solution module."""
    path = Path(solution_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def plot_territory_hull(ax, points, color, alpha=0.2):
    """Plot a convex hull around territory points."""
    if len(points) < 3:
        return  # Need at least 3 points for a hull

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        # Close the polygon
        hull_points = np.vstack([hull_points, hull_points[0]])
        ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=alpha)
        ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, linewidth=1, alpha=0.5)
    except Exception:
        pass  # ConvexHull can fail with collinear points


def visualize_territories(solution_path: str, output_path: str = None, num_reps: int = 4):
    """
    Create territory visualization.

    Args:
        solution_path: Path to solution Python file
        output_path: Output PNG path (or None to display)
        num_reps: Number of sales reps
    """
    accounts = get_accounts()
    module = load_solution(solution_path)
    assignments = module.assign_territories(accounts, num_reps)

    # Create account lookup
    account_map = {a["id"]: a for a in accounts}

    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot each territory
    for rep_id in range(num_reps):
        account_ids = assignments.get(rep_id, [])
        if not account_ids:
            continue

        rep_accounts = [account_map[aid] for aid in account_ids if aid in account_map]
        lons = [a["lon"] for a in rep_accounts]
        lats = [a["lat"] for a in rep_accounts]
        revenues = [a["revenue"] for a in rep_accounts]

        color = COLORS[rep_id % len(COLORS)]
        name = REP_NAMES[rep_id % len(REP_NAMES)]

        # Plot convex hull for territory boundary
        if len(rep_accounts) >= 3:
            points = np.array([[a["lon"], a["lat"]] for a in rep_accounts])
            plot_territory_hull(ax, points, color)

        # Plot accounts as scatter points
        # Size proportional to revenue
        sizes = [max(50, r / 2000) for r in revenues]
        total_rev = sum(revenues)
        ax.scatter(
            lons, lats, c=color, s=sizes, alpha=0.8,
            label=f"{name}: {len(account_ids)} accounts, ${total_rev:,}",
            edgecolors="white", linewidth=1
        )

        # Plot territory centroid
        centroid_lon = sum(lons) / len(lons)
        centroid_lat = sum(lats) / len(lats)
        ax.scatter(
            [centroid_lon], [centroid_lat], c=color, s=200,
            marker="*", edgecolors="black", linewidth=2, zorder=10
        )

    # Add labels for high-value accounts
    for account in accounts:
        if account["revenue"] >= 200000:
            ax.annotate(
                f"${account['revenue'] // 1000}k",
                (account["lon"], account["lat"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.7
            )

    # Styling
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Sales Territory Assignments", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add solution name to title
    solution_name = Path(solution_path).stem
    ax.set_title(f"Sales Territory Assignments - {solution_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize territory assignments")
    parser.add_argument("solution", help="Path to solution Python file")
    parser.add_argument("--output", "-o", help="Output PNG path (default: display)")
    parser.add_argument("--reps", type=int, default=4, help="Number of sales reps")

    args = parser.parse_args()

    visualize_territories(args.solution, args.output, args.reps)


if __name__ == "__main__":
    main()
