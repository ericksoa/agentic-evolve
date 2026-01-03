#!/usr/bin/env python3
"""
No-Fit Polygon based optimizer for Santa 2025 Tree Packing.

Uses NFP (No-Fit Polygon) precomputation and optimization to find
better packings than the greedy approach.

Key insight: The NFP defines where one tree's center can be placed
relative to another tree such that they don't overlap.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import json

# Tree polygon vertices (from Rust lib.rs)
TREE_VERTICES = np.array([
    (0.0, 0.8),      # Tip
    (0.125, 0.5),    # Right side - Top Tier
    (0.0625, 0.5),
    (0.2, 0.25),     # Right side - Middle Tier
    (0.1, 0.25),
    (0.35, 0.0),     # Right side - Bottom Tier
    (0.075, 0.0),    # Right Trunk
    (0.075, -0.2),
    (-0.075, -0.2),  # Left Trunk
    (-0.075, 0.0),
    (-0.35, 0.0),    # Left side - Bottom Tier
    (-0.1, 0.25),    # Left side - Middle Tier
    (-0.2, 0.25),
    (-0.0625, 0.5),  # Left side - Top Tier
    (-0.125, 0.5),
])

ROTATION_ANGLES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def rotate_polygon(vertices: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate polygon vertices by given angle (degrees)."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return vertices @ rotation_matrix.T


def translate_polygon(vertices: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate polygon vertices by (dx, dy)."""
    return vertices + np.array([dx, dy])


def polygon_bounds(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) bounding box."""
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    return min_x, min_y, max_x, max_y


def cross_product_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(a1: np.ndarray, a2: np.ndarray,
                       b1: np.ndarray, b2: np.ndarray) -> bool:
    """Check if segment (a1,a2) intersects segment (b1,b2)."""
    eps = 1e-9
    d1 = cross_product_2d(b1, b2, a1)
    d2 = cross_product_2d(b1, b2, a2)
    d3 = cross_product_2d(a1, a2, b1)
    d4 = cross_product_2d(a1, a2, b2)

    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return False


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if point is inside polygon using winding number."""
    winding = 0
    n = len(polygon)
    px, py = point

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if y1 <= py:
            if y2 > py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross > 1e-10:
                    winding += 1
        else:
            if y2 <= py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross < -1e-10:
                    winding -= 1

    return winding != 0


def polygons_overlap(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """Check if two polygons overlap (using edge intersection + containment)."""
    # Check bounding boxes first
    b1 = polygon_bounds(poly1)
    b2 = polygon_bounds(poly2)
    eps = 1e-9

    if b1[2] + eps < b2[0] or b2[2] + eps < b1[0] or \
       b1[3] + eps < b2[1] or b2[3] + eps < b1[1]:
        return False

    n1, n2 = len(poly1), len(poly2)

    # Check edge intersections
    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i + 1) % n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j + 1) % n2]
            if segments_intersect(a1, a2, b1, b2):
                return True

    # Check point containment
    for p in poly1:
        if point_in_polygon(p, poly2):
            return True
    for p in poly2:
        if point_in_polygon(p, poly1):
            return True

    return False


@dataclass
class PlacedTree:
    """A tree placed at position (x, y) with rotation angle_idx (0-7)."""
    x: float
    y: float
    angle_idx: int  # 0-7 for 45 degree increments

    def get_vertices(self) -> np.ndarray:
        """Get the transformed vertices of this tree."""
        angle = ROTATION_ANGLES[self.angle_idx]
        vertices = rotate_polygon(TREE_VERTICES, angle)
        return translate_polygon(vertices, self.x, self.y)

    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box."""
        return polygon_bounds(self.get_vertices())


def trees_overlap(tree1: PlacedTree, tree2: PlacedTree) -> bool:
    """Check if two placed trees overlap."""
    return polygons_overlap(tree1.get_vertices(), tree2.get_vertices())


def compute_packing_side(trees: List[PlacedTree]) -> float:
    """Compute the side length of the smallest enclosing square."""
    if not trees:
        return 0.0

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for tree in trees:
        bx1, by1, bx2, by2 = tree.bounds()
        min_x = min(min_x, bx1)
        min_y = min(min_y, by1)
        max_x = max(max_x, bx2)
        max_y = max(max_y, by2)

    return max(max_x - min_x, max_y - min_y)


def check_packing_valid(trees: List[PlacedTree]) -> bool:
    """Check if no trees overlap."""
    n = len(trees)
    for i in range(n):
        for j in range(i + 1, n):
            if trees_overlap(trees[i], trees[j]):
                return False
    return True


class NFPCache:
    """Cache of No-Fit Polygons for all rotation pairs."""

    def __init__(self):
        """Precompute NFPs for all 64 rotation pairs."""
        self.nfps = {}
        self._compute_all_nfps()

    def _compute_nfp(self, rot1: int, rot2: int) -> np.ndarray:
        """
        Compute the No-Fit Polygon for tree with rotation rot2
        placed relative to tree with rotation rot1.

        Uses Minkowski sum approach: NFP = P1 ⊕ (-P2)
        The reference point of P2 must be outside NFP for no overlap.
        """
        poly1 = rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[rot1])
        poly2 = rotate_polygon(TREE_VERTICES, ROTATION_ANGLES[rot2])

        # Minkowski sum approximation using boundary sampling
        # For non-convex polygons, we sample the boundary
        samples = []

        # Sample positions around poly1 where poly2's center can be
        # For each edge of poly1, slide poly2 along it
        n1 = len(poly1)
        n2 = len(poly2)

        # More thorough: for each vertex of each polygon, compute offset
        for v1 in poly1:
            for v2 in poly2:
                # Position where poly2's origin would be if v2 touches v1
                offset = v1 - v2
                samples.append(offset)

        # Also sample along edges
        for i in range(n1):
            e1_start = poly1[i]
            e1_end = poly1[(i + 1) % n1]
            for t in np.linspace(0, 1, 5):
                pt = e1_start + t * (e1_end - e1_start)
                for v2 in poly2:
                    offset = pt - v2
                    samples.append(offset)

        samples = np.array(samples)

        # Compute convex hull of samples as approximation
        # For non-convex polygons, this is an over-approximation
        if len(samples) >= 3:
            try:
                hull = ConvexHull(samples)
                return samples[hull.vertices]
            except:
                return samples
        return samples

    def _compute_all_nfps(self):
        """Compute all 64 NFPs."""
        for r1 in range(8):
            for r2 in range(8):
                self.nfps[(r1, r2)] = self._compute_nfp(r1, r2)

    def get_min_distance(self, rot1: int, rot2: int,
                        direction: np.ndarray) -> float:
        """
        Get minimum distance in given direction where tree2 can be placed
        relative to tree1 without overlap.
        """
        nfp = self.nfps[(rot1, rot2)]
        direction = direction / np.linalg.norm(direction)

        # Find the furthest NFP point in this direction
        projections = nfp @ direction
        return np.max(projections)


def optimize_packing_scipy(n: int, max_iterations: int = 1000) -> List[PlacedTree]:
    """
    Use scipy differential evolution to find a good packing for n trees.
    """
    if n == 1:
        return [PlacedTree(0.0, 0.0, 0)]

    # Variables: x1, y1, r1, x2, y2, r2, ... for n trees
    # First tree fixed at origin
    n_vars = (n - 1) * 2  # Only positions of trees 2..n

    # Rotation assignments (discrete, try multiple)
    best_trees = None
    best_side = float('inf')

    # Try different rotation assignments
    for rotation_attempt in range(min(20, 8 ** min(n, 3))):
        # Generate rotation assignment
        rotations = [0]  # First tree at rotation 0
        for i in range(1, n):
            rotations.append((rotation_attempt + i) % 8)

        def objective(x):
            """Minimize bounding box side with overlap penalty."""
            trees = [PlacedTree(0.0, 0.0, rotations[0])]
            for i in range(n - 1):
                trees.append(PlacedTree(x[2*i], x[2*i + 1], rotations[i + 1]))

            side = compute_packing_side(trees)

            # Add overlap penalty
            penalty = 0.0
            for i in range(len(trees)):
                for j in range(i + 1, len(trees)):
                    if trees_overlap(trees[i], trees[j]):
                        penalty += 10.0

            return side + penalty

        # Bounds for positions
        bounds = [(-3.0, 3.0)] * n_vars

        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations // 20,
            seed=rotation_attempt,
            polish=True,
            updating='immediate',
            workers=1,  # Single-threaded to avoid pickling issues
            tol=1e-4
        )

        # Reconstruct solution
        trees = [PlacedTree(0.0, 0.0, rotations[0])]
        for i in range(n - 1):
            trees.append(PlacedTree(result.x[2*i], result.x[2*i + 1], rotations[i + 1]))

        if check_packing_valid(trees):
            side = compute_packing_side(trees)
            if side < best_side:
                best_side = side
                best_trees = trees

    if best_trees is None:
        # Fallback: simple spiral placement
        best_trees = simple_spiral_placement(n)

    return best_trees


def simple_spiral_placement(n: int) -> List[PlacedTree]:
    """Simple spiral placement as fallback."""
    trees = []
    golden_angle = np.pi * (3 - np.sqrt(5))

    for i in range(n):
        if i == 0:
            trees.append(PlacedTree(0.0, 0.0, 0))
        else:
            angle = i * golden_angle
            # Find valid position
            for dist in np.linspace(0.3, 6.0, 100):
                x = dist * np.cos(angle)
                y = dist * np.sin(angle)

                for rot in range(8):
                    tree = PlacedTree(x, y, rot)
                    valid = True
                    for existing in trees:
                        if trees_overlap(tree, existing):
                            valid = False
                            break
                    if valid:
                        trees.append(tree)
                        break
                else:
                    continue
                break
            else:
                # Couldn't place, use a far position
                trees.append(PlacedTree(6.0 * np.cos(angle), 6.0 * np.sin(angle), 0))

    return trees


def greedy_nfp_placement(n: int, nfp_cache: NFPCache) -> List[PlacedTree]:
    """
    Greedy placement using NFP-guided positioning.
    Places each tree as close to center as possible.
    """
    if n == 0:
        return []

    trees = [PlacedTree(0.0, 0.0, 0)]

    for i in range(1, n):
        best_tree = None
        best_side = float('inf')

        # Try many directions
        for dir_idx in range(32):
            angle = dir_idx * 2 * np.pi / 32
            direction = np.array([np.cos(angle), np.sin(angle)])

            # Try all rotations
            for rot in range(8):
                # Compute minimum distance from all existing trees
                min_dist = 0.0
                for existing in trees:
                    # Get NFP-based distance
                    nfp_dist = nfp_cache.get_min_distance(existing.angle_idx, rot, direction)

                    # Offset from existing tree position
                    proj = existing.x * direction[0] + existing.y * direction[1]
                    required_dist = proj + nfp_dist + 0.02  # Small buffer
                    min_dist = max(min_dist, required_dist)

                # Place tree at computed distance
                x = min_dist * direction[0]
                y = min_dist * direction[1]
                tree = PlacedTree(x, y, rot)

                # Verify no overlap
                valid = True
                for existing in trees:
                    if trees_overlap(tree, existing):
                        valid = False
                        break

                if valid:
                    test_trees = trees + [tree]
                    side = compute_packing_side(test_trees)
                    if side < best_side:
                        best_side = side
                        best_tree = tree

        if best_tree:
            trees.append(best_tree)
        else:
            # Fallback
            trees.append(PlacedTree(3.0, 3.0, 0))

    return trees


def local_search(trees: List[PlacedTree], iterations: int = 1000) -> List[PlacedTree]:
    """Local search to improve a packing."""
    current = [PlacedTree(t.x, t.y, t.angle_idx) for t in trees]
    current_side = compute_packing_side(current)

    best = current
    best_side = current_side

    for _ in range(iterations):
        # Pick random tree to move
        idx = np.random.randint(len(current))
        old = current[idx]

        # Try small perturbation
        dx = np.random.normal(0, 0.05)
        dy = np.random.normal(0, 0.05)
        new_rot = (old.angle_idx + np.random.choice([-1, 0, 1])) % 8

        current[idx] = PlacedTree(old.x + dx, old.y + dy, new_rot)

        if check_packing_valid(current):
            new_side = compute_packing_side(current)
            if new_side < current_side:
                current_side = new_side
                if new_side < best_side:
                    best = [PlacedTree(t.x, t.y, t.angle_idx) for t in current]
                    best_side = new_side
            elif np.random.random() < 0.1:  # Accept with small probability
                current_side = new_side
            else:
                current[idx] = old
        else:
            current[idx] = old

    return best


def analyze_small_packings(max_n: int = 10):
    """Analyze optimal packings for small n to discover patterns."""
    nfp_cache = NFPCache()

    results = {}

    for n in range(1, max_n + 1):
        print(f"\n=== Optimizing n={n} ===")

        # Try multiple approaches
        approaches = []

        # 1. NFP-guided greedy
        trees = greedy_nfp_placement(n, nfp_cache)
        trees = local_search(trees, iterations=500)
        side = compute_packing_side(trees)
        approaches.append(('nfp_greedy', side, trees))
        print(f"NFP greedy: side = {side:.4f}")

        # 2. Scipy optimization (for small n)
        if n <= 6:
            trees2 = optimize_packing_scipy(n, max_iterations=500)
            trees2 = local_search(trees2, iterations=500)
            side2 = compute_packing_side(trees2)
            approaches.append(('scipy', side2, trees2))
            print(f"Scipy: side = {side2:.4f}")

        # 3. Simple spiral
        trees3 = simple_spiral_placement(n)
        trees3 = local_search(trees3, iterations=500)
        side3 = compute_packing_side(trees3)
        approaches.append(('spiral', side3, trees3))
        print(f"Spiral: side = {side3:.4f}")

        # Select best
        best_name, best_side, best_trees = min(approaches, key=lambda x: x[1])
        print(f"Best: {best_name} with side = {best_side:.4f}")

        # Verify validity
        if not check_packing_valid(best_trees):
            print("WARNING: Invalid packing!")

        results[n] = {
            'side': float(best_side),
            'trees': [(float(t.x), float(t.y), int(t.angle_idx)) for t in best_trees],
            'method': best_name
        }

        # Print tree positions for pattern analysis
        print("Positions:")
        for i, t in enumerate(best_trees):
            print(f"  Tree {i}: ({t.x:.4f}, {t.y:.4f}) rot={t.angle_idx * 45}°")

    return results


def export_to_rust_format(results: dict, filename: str):
    """Export optimized packings to a format Rust can read."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Exported to {filename}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NFP-based tree packing optimizer')
    parser.add_argument('--max-n', type=int, default=10, help='Maximum n to optimize')
    parser.add_argument('--output', type=str, default='optimized_packings.json', help='Output file')
    args = parser.parse_args()

    print("Santa 2025 Tree Packing - NFP Optimizer")
    print("=" * 50)

    results = analyze_small_packings(args.max_n)

    # Calculate partial score
    score = sum(r['side']**2 / n for n, r in results.items())
    print(f"\nPartial score (n=1..{args.max_n}): {score:.4f}")

    export_to_rust_format(results, args.output)
