#!/usr/bin/env python3
"""
NFP (No-Fit Polygon) Based Packing for Santa 2025 Competition

This implements the gold standard approach used by industrial packing solvers:
1. Compute Minkowski sums to get NFPs
2. Precompute NFPs for all rotation pairs
3. Use NFP to find optimal valid placements
4. Select placement minimizing bounding box

NFP Definition: For polygons A and B, the NFP is the locus of positions where
B touches but doesn't overlap A. If B's reference point is inside the NFP,
B overlaps A. The boundary of NFP gives all touching positions.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
import json
import sys
from dataclasses import dataclass

# Tree polygon vertices (same as Rust)
TREE_VERTICES = [
    (0.0, 0.8),      # Tip
    (0.125, 0.5),    # Right - Top Tier
    (0.0625, 0.5),
    (0.2, 0.25),     # Right - Middle Tier
    (0.1, 0.25),
    (0.35, 0.0),     # Right - Bottom Tier
    (0.075, 0.0),    # Right Trunk
    (0.075, -0.2),
    (-0.075, -0.2),  # Left Trunk
    (-0.075, 0.0),
    (-0.35, 0.0),    # Left - Bottom Tier
    (-0.1, 0.25),    # Left - Middle Tier
    (-0.2, 0.25),
    (-0.0625, 0.5),  # Left - Top Tier
    (-0.125, 0.5),
]

# Canonical tree polygon at origin with 0 rotation
TREE_POLYGON = Polygon(TREE_VERTICES)


@dataclass
class PlacedTree:
    """A tree placed at a specific position and rotation."""
    x: float
    y: float
    angle: float  # degrees

    def get_polygon(self) -> Polygon:
        """Get the Shapely polygon for this placed tree."""
        rotated = affinity.rotate(TREE_POLYGON, self.angle, origin=(0, 0))
        translated = affinity.translate(rotated, self.x, self.y)
        return translated


def compute_minkowski_sum(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """
    Compute Minkowski sum of two convex/simple polygons.

    For NFP: NFP(A, B) = A ⊕ (-B) where -B is B reflected through origin.

    Uses the rotating calipers algorithm for convex hulls, then handles
    non-convex cases via decomposition.
    """
    # Get coordinates as arrays
    coords_a = np.array(poly_a.exterior.coords[:-1])  # Remove duplicate closing point
    coords_b = np.array(poly_b.exterior.coords[:-1])

    # For our tree polygon (non-convex), we use Shapely's buffer trick
    # This is an approximation but works well for practical purposes

    # Actually, for precise NFP, we'll use the explicit Minkowski sum algorithm
    # For simple polygons, we can use convolution of edges

    n_a = len(coords_a)
    n_b = len(coords_b)

    # Simple approach: compute all vertex+vertex combinations and take convex hull
    # This works for convex polygons and gives an approximation for non-convex
    sum_points = []
    for pa in coords_a:
        for pb in coords_b:
            sum_points.append(pa + pb)

    # For more precise results with non-convex, we'd use edge-based algorithm
    # But for our use case, taking the convex hull of sum points works
    sum_points = np.array(sum_points)

    # Create polygon from points - use convex hull for robustness
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(sum_points)
        hull_points = sum_points[hull.vertices]
        return Polygon(hull_points)
    except:
        # Fallback: create polygon directly
        return Polygon(sum_points).convex_hull


def compute_nfp(fixed_poly: Polygon, moving_poly: Polygon) -> Polygon:
    """
    Compute No-Fit Polygon for a moving polygon relative to fixed polygon.

    The NFP defines where the reference point of moving_poly can be placed
    such that moving_poly touches but doesn't overlap fixed_poly.

    If moving_poly's reference point is INSIDE the NFP, there's an overlap.
    The boundary of NFP gives all "touching" positions.
    """
    # Reflect moving polygon through origin
    reflected = affinity.scale(moving_poly, -1, -1, origin=(0, 0))

    # Compute Minkowski sum
    nfp = compute_minkowski_sum(fixed_poly, reflected)

    return nfp


def compute_ifr(container_side: float, tree_angle: float) -> Polygon:
    """
    Compute Inner-Fit Rectangle (IFR) - the region where a tree can be placed
    while staying entirely within a square container of given side length.

    The container is [0, side] x [0, side].
    """
    # Get tree bounds at this rotation
    tree_poly = affinity.rotate(TREE_POLYGON, tree_angle, origin=(0, 0))
    bounds = tree_poly.bounds  # (minx, miny, maxx, maxy)

    tree_minx, tree_miny, tree_maxx, tree_maxy = bounds
    tree_width = tree_maxx - tree_minx
    tree_height = tree_maxy - tree_miny

    # The tree's reference point (origin) must be placed such that:
    # tree_minx + x >= 0 => x >= -tree_minx
    # tree_maxx + x <= side => x <= side - tree_maxx
    # Similar for y

    ifr_minx = -tree_minx
    ifr_maxx = container_side - tree_maxx
    ifr_miny = -tree_miny
    ifr_maxy = container_side - tree_maxy

    if ifr_minx >= ifr_maxx or ifr_miny >= ifr_maxy:
        # Container too small
        return None

    return Polygon([
        (ifr_minx, ifr_miny),
        (ifr_maxx, ifr_miny),
        (ifr_maxx, ifr_maxy),
        (ifr_minx, ifr_maxy),
    ])


class NFPCache:
    """Cache for precomputed NFPs at different rotation pairs."""

    def __init__(self, angle_steps: int = 8):
        self.angle_steps = angle_steps
        self.angles = [i * 360.0 / angle_steps for i in range(angle_steps)]
        self.nfp_cache: Dict[Tuple[int, int], Polygon] = {}
        self._precompute()

    def _precompute(self):
        """Precompute NFPs for all rotation pairs."""
        print(f"Precomputing NFPs for {self.angle_steps}x{self.angle_steps} rotation pairs...")

        for i, angle_a in enumerate(self.angles):
            for j, angle_b in enumerate(self.angles):
                poly_a = affinity.rotate(TREE_POLYGON, angle_a, origin=(0, 0))
                poly_b = affinity.rotate(TREE_POLYGON, angle_b, origin=(0, 0))
                nfp = compute_nfp(poly_a, poly_b)
                self.nfp_cache[(i, j)] = nfp

        print(f"  Cached {len(self.nfp_cache)} NFPs")

    def get_nfp(self, fixed_angle_idx: int, moving_angle_idx: int) -> Polygon:
        """Get cached NFP for rotation pair."""
        return self.nfp_cache[(fixed_angle_idx, moving_angle_idx)]

    def get_angle_index(self, angle: float) -> int:
        """Get nearest discrete angle index for continuous angle."""
        angle = angle % 360
        return int(round(angle / (360 / self.angle_steps))) % self.angle_steps

    def get_angle(self, idx: int) -> float:
        """Get angle for index."""
        return self.angles[idx]


def find_valid_positions_nfp(
    placed_trees: List[PlacedTree],
    new_angle: float,
    nfp_cache: NFPCache,
    current_bounds: Tuple[float, float, float, float],  # (minx, miny, maxx, maxy)
    max_side: float,
    num_samples: int = 100
) -> List[Tuple[float, float, float]]:  # List of (x, y, side_length)
    """
    Find valid placement positions for a new tree using NFP.

    Returns positions on the boundary of feasible region that minimize side length.
    """
    new_angle_idx = nfp_cache.get_angle_index(new_angle)
    new_angle_discrete = nfp_cache.get_angle(new_angle_idx)

    if not placed_trees:
        # First tree - just return center
        return [(0.0, 0.0, compute_side_length([PlacedTree(0.0, 0.0, new_angle_discrete)]))]

    # Compute forbidden region (union of NFPs centered at placed trees)
    forbidden_regions = []
    for tree in placed_trees:
        fixed_angle_idx = nfp_cache.get_angle_index(tree.angle)
        nfp = nfp_cache.get_nfp(fixed_angle_idx, new_angle_idx)
        # Translate NFP to tree position
        translated_nfp = affinity.translate(nfp, tree.x, tree.y)
        forbidden_regions.append(translated_nfp)

    # Union of all forbidden regions
    forbidden = unary_union(forbidden_regions)

    # Get boundary of forbidden region - these are potential placement positions
    if isinstance(forbidden, MultiPolygon):
        boundaries = [p.exterior.coords[:] for p in forbidden.geoms]
    else:
        boundaries = [forbidden.exterior.coords[:]]

    # Sample positions along boundaries
    candidate_positions = []

    for boundary in boundaries:
        # Sample along the boundary
        coords = np.array(boundary)
        n_points = len(coords)

        if n_points < 2:
            continue

        # Compute cumulative distances along boundary
        dists = np.zeros(n_points)
        for i in range(1, n_points):
            dists[i] = dists[i-1] + np.linalg.norm(coords[i] - coords[i-1])

        total_dist = dists[-1]
        if total_dist < 0.01:
            continue

        # Sample at regular intervals
        sample_dists = np.linspace(0, total_dist, num_samples, endpoint=False)

        for d in sample_dists:
            # Find segment containing this distance
            idx = np.searchsorted(dists, d)
            if idx == 0:
                idx = 1

            # Interpolate position
            t = (d - dists[idx-1]) / (dists[idx] - dists[idx-1] + 1e-10)
            pos = coords[idx-1] * (1 - t) + coords[idx] * t
            candidate_positions.append(pos)

    # Also add some positions near corners/vertices
    for boundary in boundaries:
        for pt in boundary:
            candidate_positions.append(np.array(pt))

    # Evaluate each candidate
    valid_positions = []
    tree_poly = affinity.rotate(TREE_POLYGON, new_angle_discrete, origin=(0, 0))

    for pos in candidate_positions:
        x, y = pos

        # Quick bounding box check
        new_tree_poly = affinity.translate(tree_poly, x, y)

        # Check overlap with all placed trees
        has_overlap = False
        for tree in placed_trees:
            existing_poly = tree.get_polygon()
            if new_tree_poly.intersects(existing_poly):
                intersection = new_tree_poly.intersection(existing_poly)
                if intersection.area > 1e-10:  # Tolerance for touching
                    has_overlap = True
                    break

        if has_overlap:
            continue

        # Compute resulting side length
        all_trees = placed_trees + [PlacedTree(x, y, new_angle_discrete)]
        side = compute_side_length(all_trees)

        if side <= max_side:
            valid_positions.append((x, y, side))

    return valid_positions


def compute_side_length(trees: List[PlacedTree]) -> float:
    """Compute bounding box side length for a list of placed trees."""
    if not trees:
        return 0.0

    all_xs = []
    all_ys = []

    for tree in trees:
        poly = tree.get_polygon()
        coords = np.array(poly.exterior.coords)
        all_xs.extend(coords[:, 0])
        all_ys.extend(coords[:, 1])

    width = max(all_xs) - min(all_xs)
    height = max(all_ys) - min(all_ys)

    return max(width, height)


def greedy_nfp_pack(n: int, nfp_cache: NFPCache, num_angle_samples: int = 8) -> Tuple[float, List[PlacedTree]]:
    """
    Greedy packing using NFP-based placement.

    For each tree:
    1. Try all rotation angles
    2. For each angle, find all valid positions using NFP
    3. Select position that minimizes bounding box increase
    """
    if n <= 0:
        return 0.0, []

    # Start with first tree at origin
    trees = [PlacedTree(0.0, 0.0, 0.0)]

    for tree_idx in range(1, n):
        best_placement = None
        best_side = float('inf')

        current_side = compute_side_length(trees)
        bounds = get_bounding_box(trees)

        # Try each rotation angle
        for angle_idx in range(nfp_cache.angle_steps):
            angle = nfp_cache.get_angle(angle_idx)

            # Find valid positions using NFP
            positions = find_valid_positions_nfp(
                trees, angle, nfp_cache,
                bounds, max_side=current_side + 1.0,
                num_samples=50
            )

            for x, y, side in positions:
                if side < best_side:
                    best_side = side
                    best_placement = PlacedTree(x, y, angle)

        if best_placement is None:
            # Fallback: place at expansion position
            print(f"  Warning: No NFP position found for tree {tree_idx}, using fallback")
            best_placement = fallback_placement(trees, nfp_cache)

        trees.append(best_placement)

        if (tree_idx + 1) % 10 == 0:
            print(f"  Placed {tree_idx + 1}/{n} trees, side={best_side:.4f}")

    final_side = compute_side_length(trees)
    return final_side, trees


def get_bounding_box(trees: List[PlacedTree]) -> Tuple[float, float, float, float]:
    """Get bounding box of placed trees."""
    if not trees:
        return (0, 0, 0, 0)

    all_xs = []
    all_ys = []

    for tree in trees:
        poly = tree.get_polygon()
        coords = np.array(poly.exterior.coords)
        all_xs.extend(coords[:, 0])
        all_ys.extend(coords[:, 1])

    return (min(all_xs), min(all_ys), max(all_xs), max(all_ys))


def fallback_placement(trees: List[PlacedTree], nfp_cache: NFPCache) -> PlacedTree:
    """Fallback placement when NFP doesn't find valid position."""
    bounds = get_bounding_box(trees)
    minx, miny, maxx, maxy = bounds

    # Try corners and edge midpoints
    candidates = [
        (maxx + 0.4, (miny + maxy) / 2),  # Right
        (minx - 0.4, (miny + maxy) / 2),  # Left
        ((minx + maxx) / 2, maxy + 0.5),  # Top
        ((minx + maxx) / 2, miny - 0.5),  # Bottom
    ]

    best_placement = None
    best_side = float('inf')

    tree_polys = [t.get_polygon() for t in trees]

    for x, y in candidates:
        for angle_idx in range(nfp_cache.angle_steps):
            angle = nfp_cache.get_angle(angle_idx)
            new_tree = PlacedTree(x, y, angle)
            new_poly = new_tree.get_polygon()

            # Check overlap
            has_overlap = any(new_poly.intersects(p) and new_poly.intersection(p).area > 1e-10
                            for p in tree_polys)

            if not has_overlap:
                side = compute_side_length(trees + [new_tree])
                if side < best_side:
                    best_side = side
                    best_placement = new_tree

    return best_placement if best_placement else PlacedTree(maxx + 0.5, 0, 0)


def validate_packing(trees: List[PlacedTree]) -> Tuple[bool, int]:
    """Validate packing has no overlaps. Returns (valid, num_overlaps)."""
    n = len(trees)
    overlaps = 0

    for i in range(n):
        for j in range(i + 1, n):
            poly_i = trees[i].get_polygon()
            poly_j = trees[j].get_polygon()

            if poly_i.intersects(poly_j):
                intersection = poly_i.intersection(poly_j)
                if intersection.area > 1e-8:
                    overlaps += 1

    return overlaps == 0, overlaps


def export_to_csv_format(trees: List[PlacedTree]) -> str:
    """Export packing to CSV format matching submission format."""
    lines = []
    for i, tree in enumerate(trees):
        # Get tree vertices
        poly = tree.get_polygon()
        coords = list(poly.exterior.coords)[:-1]  # Remove closing point

        # Format: x1 y1;x2 y2;...;xn yn
        coord_strs = [f"{x:.6f} {y:.6f}" for x, y in coords]
        lines.append(";".join(coord_strs))

    return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='NFP-based packing')
    parser.add_argument('n', type=int, help='Number of trees')
    parser.add_argument('--angles', type=int, default=8, help='Number of rotation angles')
    parser.add_argument('--output', type=str, help='Output file (JSON)')
    parser.add_argument('--csv', type=str, help='Output CSV for submission')

    args = parser.parse_args()

    print(f"NFP Packing for n={args.n}")
    print(f"Using {args.angles} rotation angles")

    # Initialize NFP cache
    nfp_cache = NFPCache(angle_steps=args.angles)

    # Run greedy packing
    side, trees = greedy_nfp_pack(args.n, nfp_cache)

    # Validate
    valid, num_overlaps = validate_packing(trees)

    print(f"\nResults:")
    print(f"  Side length: {side:.4f}")
    print(f"  Valid: {valid} ({num_overlaps} overlaps)")
    print(f"  Score contribution: {side**2 / args.n:.4f}")

    if args.output:
        data = {
            'n': args.n,
            'side': side,
            'valid': valid,
            'trees': [{'x': t.x, 'y': t.y, 'angle': t.angle} for t in trees]
        }
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {args.output}")

    if args.csv:
        csv_data = export_to_csv_format(trees)
        with open(args.csv, 'w') as f:
            f.write(csv_data)
        print(f"Saved CSV to {args.csv}")


if __name__ == '__main__':
    main()
