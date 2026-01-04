#!/usr/bin/env python3
"""
Gen107: Numba-accelerated polygon collision detection.

This module provides fast polygon overlap detection using Numba JIT compilation.
The algorithm matches the Rust implementation in lib.rs:
1. BBox check (fast filter)
2. Edge intersection detection (segments_intersect_proper)
3. Point-in-polygon (winding number)
4. Safety margin check

Target: Match Rust collision speed for CPU-bound polygon checks.
"""

import numpy as np
import numba
from numba import njit, prange
import time
from typing import Tuple

# Tree shape constants (15 vertices) - matches Rust TREE_VERTICES
TREE_VERTICES = np.array([
    [0.0, 0.8],       # Tip
    [0.125, 0.5],     # Right top tier
    [0.0625, 0.5],
    [0.2, 0.25],      # Right mid tier
    [0.1, 0.25],
    [0.35, 0.0],      # Right bottom tier
    [0.075, 0.0],     # Right trunk
    [0.075, -0.2],
    [-0.075, -0.2],   # Left trunk
    [-0.075, 0.0],
    [-0.35, 0.0],     # Left bottom tier
    [-0.1, 0.25],     # Left mid tier
    [-0.2, 0.25],
    [-0.0625, 0.5],   # Left top tier
    [-0.125, 0.5],
], dtype=np.float64)


# Inline tree vertices as tuples for Numba performance
# Numba handles tuples of literals much faster than global arrays
_TREE_VX = (0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125)
_TREE_VY = (0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5)


@njit(cache=True)
def transform_tree(x: float, y: float, angle_deg: float) -> np.ndarray:
    """Transform base tree vertices by rotation and translation."""
    angle_rad = angle_deg * (np.pi / 180.0)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    result = np.empty((15, 2), dtype=np.float64)
    # Unrolled loop with inline vertex values for Numba performance
    result[0, 0] = 0.0 * cos_a - 0.8 * sin_a + x
    result[0, 1] = 0.0 * sin_a + 0.8 * cos_a + y
    result[1, 0] = 0.125 * cos_a - 0.5 * sin_a + x
    result[1, 1] = 0.125 * sin_a + 0.5 * cos_a + y
    result[2, 0] = 0.0625 * cos_a - 0.5 * sin_a + x
    result[2, 1] = 0.0625 * sin_a + 0.5 * cos_a + y
    result[3, 0] = 0.2 * cos_a - 0.25 * sin_a + x
    result[3, 1] = 0.2 * sin_a + 0.25 * cos_a + y
    result[4, 0] = 0.1 * cos_a - 0.25 * sin_a + x
    result[4, 1] = 0.1 * sin_a + 0.25 * cos_a + y
    result[5, 0] = 0.35 * cos_a - 0.0 * sin_a + x
    result[5, 1] = 0.35 * sin_a + 0.0 * cos_a + y
    result[6, 0] = 0.075 * cos_a - 0.0 * sin_a + x
    result[6, 1] = 0.075 * sin_a + 0.0 * cos_a + y
    result[7, 0] = 0.075 * cos_a - (-0.2) * sin_a + x
    result[7, 1] = 0.075 * sin_a + (-0.2) * cos_a + y
    result[8, 0] = -0.075 * cos_a - (-0.2) * sin_a + x
    result[8, 1] = -0.075 * sin_a + (-0.2) * cos_a + y
    result[9, 0] = -0.075 * cos_a - 0.0 * sin_a + x
    result[9, 1] = -0.075 * sin_a + 0.0 * cos_a + y
    result[10, 0] = -0.35 * cos_a - 0.0 * sin_a + x
    result[10, 1] = -0.35 * sin_a + 0.0 * cos_a + y
    result[11, 0] = -0.1 * cos_a - 0.25 * sin_a + x
    result[11, 1] = -0.1 * sin_a + 0.25 * cos_a + y
    result[12, 0] = -0.2 * cos_a - 0.25 * sin_a + x
    result[12, 1] = -0.2 * sin_a + 0.25 * cos_a + y
    result[13, 0] = -0.0625 * cos_a - 0.5 * sin_a + x
    result[13, 1] = -0.0625 * sin_a + 0.5 * cos_a + y
    result[14, 0] = -0.125 * cos_a - 0.5 * sin_a + x
    result[14, 1] = -0.125 * sin_a + 0.5 * cos_a + y

    return result


@njit(cache=True)
def polygon_bounds(poly: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute bounding box of polygon."""
    min_x = poly[0, 0]
    min_y = poly[0, 1]
    max_x = poly[0, 0]
    max_y = poly[0, 1]

    for i in range(1, len(poly)):
        x, y = poly[i, 0], poly[i, 1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return min_x, min_y, max_x, max_y


@njit(cache=True)
def cross_product_sign(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Compute cross product (b - a) x (c - a)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def point_near_segment(
    px: float, py: float,
    s1x: float, s1y: float,
    s2x: float, s2y: float,
    epsilon: float
) -> bool:
    """Check if point p is within epsilon of segment (s1, s2)."""
    dx = s2x - s1x
    dy = s2y - s1y
    len_sq = dx * dx + dy * dy

    if len_sq < epsilon * epsilon:
        # Degenerate segment
        d = np.sqrt((px - s1x) ** 2 + (py - s1y) ** 2)
        return d < epsilon

    # Project p onto line, get parameter t
    t = ((px - s1x) * dx + (py - s1y) * dy) / len_sq

    # Check if projection is within segment
    if t < -epsilon or t > 1.0 + epsilon:
        return False

    # Calculate distance from p to closest point on segment
    t_clamped = max(0.0, min(1.0, t))
    closest_x = s1x + t_clamped * dx
    closest_y = s1y + t_clamped * dy
    dist = np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    return dist < epsilon


@njit(cache=True)
def segments_touch_or_overlap(
    a1x: float, a1y: float, a2x: float, a2y: float,
    b1x: float, b1y: float, b2x: float, b2y: float
) -> bool:
    """Check if segments touch or overlap (for collinear/near-collinear cases)."""
    EPSILON = 1e-6

    # Check if any endpoint of one segment is very close to the other segment
    if point_near_segment(a1x, a1y, b1x, b1y, b2x, b2y, EPSILON):
        return True
    if point_near_segment(a2x, a2y, b1x, b1y, b2x, b2y, EPSILON):
        return True
    if point_near_segment(b1x, b1y, a1x, a1y, a2x, a2y, EPSILON):
        return True
    if point_near_segment(b2x, b2y, a1x, a1y, a2x, a2y, EPSILON):
        return True
    return False


@njit(cache=True)
def segments_intersect_proper(
    a1x: float, a1y: float, a2x: float, a2y: float,
    b1x: float, b1y: float, b2x: float, b2y: float
) -> bool:
    """Check if two segments intersect (including touching/collinear cases)."""
    EPSILON = 1e-6

    d1 = cross_product_sign(b1x, b1y, b2x, b2y, a1x, a1y)
    d2 = cross_product_sign(b1x, b1y, b2x, b2y, a2x, a2y)
    d3 = cross_product_sign(a1x, a1y, a2x, a2y, b1x, b1y)
    d4 = cross_product_sign(a1x, a1y, a2x, a2y, b2x, b2y)

    # Proper intersection: opposite signs
    if ((d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)):
        if ((d3 > EPSILON and d4 < -EPSILON) or (d3 < -EPSILON and d4 > EPSILON)):
            return True

    # Also check for collinear/touching cases (conservative)
    if abs(d1) < EPSILON or abs(d2) < EPSILON or abs(d3) < EPSILON or abs(d4) < EPSILON:
        return segments_touch_or_overlap(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y)

    return False


@njit(cache=True)
def point_strictly_inside_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """Check if point is strictly inside polygon (not on boundary) using winding number."""
    EPSILON = 1e-6
    winding = 0
    n = len(poly)

    for i in range(n):
        j = (i + 1) % n
        x1, y1 = poly[i, 0], poly[i, 1]
        x2, y2 = poly[j, 0], poly[j, 1]

        if y1 <= py:
            if y2 > py:
                # Upward crossing
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross > EPSILON:
                    winding += 1
        elif y2 <= py:
            # Downward crossing
            cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
            if cross < -EPSILON:
                winding -= 1

    return winding != 0


@njit(cache=True)
def polygons_overlap(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    Check if two polygons overlap or are too close.

    This matches the Rust implementation in lib.rs:
    1. BBox check (fast filter)
    2. Edge intersection detection
    3. Point-in-polygon check
    4. Safety margin check

    Args:
        poly1: (n1, 2) array of polygon vertices
        poly2: (n2, 2) array of polygon vertices

    Returns:
        True if polygons overlap or are within safety margin
    """
    SAFETY_MARGIN = 1e-5
    EPSILON = 1e-9

    # 1. BBox check (fast filter)
    min1x, min1y, max1x, max1y = polygon_bounds(poly1)
    min2x, min2y, max2x, max2y = polygon_bounds(poly2)

    if max1x + EPSILON < min2x or max2x + EPSILON < min1x:
        return False
    if max1y + EPSILON < min2y or max2y + EPSILON < min1y:
        return False

    n1 = len(poly1)
    n2 = len(poly2)

    # 2. Edge intersection check
    for i in range(n1):
        j = (i + 1) % n1
        a1x, a1y = poly1[i, 0], poly1[i, 1]
        a2x, a2y = poly1[j, 0], poly1[j, 1]

        for k in range(n2):
            l = (k + 1) % n2
            b1x, b1y = poly2[k, 0], poly2[k, 1]
            b2x, b2y = poly2[l, 0], poly2[l, 1]

            if segments_intersect_proper(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
                return True

    # 3. Point containment check
    for i in range(n1):
        if point_strictly_inside_polygon(poly1[i, 0], poly1[i, 1], poly2):
            return True

    for i in range(n2):
        if point_strictly_inside_polygon(poly2[i, 0], poly2[i, 1], poly1):
            return True

    # 4. Safety margin check
    for i in range(n1):
        px, py = poly1[i, 0], poly1[i, 1]
        for k in range(n2):
            l = (k + 1) % n2
            if point_near_segment(px, py, poly2[k, 0], poly2[k, 1],
                                  poly2[l, 0], poly2[l, 1], SAFETY_MARGIN):
                return True

    for i in range(n2):
        px, py = poly2[i, 0], poly2[i, 1]
        for k in range(n1):
            l = (k + 1) % n1
            if point_near_segment(px, py, poly1[k, 0], poly1[k, 1],
                                  poly1[l, 0], poly1[l, 1], SAFETY_MARGIN):
                return True

    return False


@njit(cache=True)
def polygons_overlap_fast(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    Fast overlap check without safety margin.
    Used for SA where we want faster iteration and only check margin at the end.
    """
    EPSILON = 1e-9

    # 1. BBox check (fast filter)
    min1x, min1y, max1x, max1y = polygon_bounds(poly1)
    min2x, min2y, max2x, max2y = polygon_bounds(poly2)

    if max1x + EPSILON < min2x or max2x + EPSILON < min1x:
        return False
    if max1y + EPSILON < min2y or max2y + EPSILON < min1y:
        return False

    n1 = len(poly1)
    n2 = len(poly2)

    # 2. Edge intersection check
    for i in range(n1):
        j = (i + 1) % n1
        a1x, a1y = poly1[i, 0], poly1[i, 1]
        a2x, a2y = poly1[j, 0], poly1[j, 1]

        for k in range(n2):
            l = (k + 1) % n2
            b1x, b1y = poly2[k, 0], poly2[k, 1]
            b2x, b2y = poly2[l, 0], poly2[l, 1]

            if segments_intersect_proper(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
                return True

    # 3. Point containment check
    for i in range(n1):
        if point_strictly_inside_polygon(poly1[i, 0], poly1[i, 1], poly2):
            return True

    for i in range(n2):
        if point_strictly_inside_polygon(poly2[i, 0], poly2[i, 1], poly1):
            return True

    return False


@njit(cache=True, parallel=True)
def count_overlaps_batch(
    vertices: np.ndarray,  # (n_trees, 15, 2)
) -> int:
    """
    Count total overlapping pairs in a single configuration.

    Args:
        vertices: (n_trees, 15, 2) transformed tree vertices

    Returns:
        Number of overlapping pairs
    """
    n_trees = vertices.shape[0]
    count = 0

    for i in prange(n_trees):
        for j in range(i + 1, n_trees):
            if polygons_overlap_fast(vertices[i], vertices[j]):
                count += 1

    return count


@njit(cache=True)
def check_tree_overlaps_any(
    new_vertices: np.ndarray,  # (15, 2) - single tree
    existing_vertices: np.ndarray,  # (n_existing, 15, 2)
) -> bool:
    """
    Check if a new tree overlaps with any existing tree.
    Used for incremental placement checking.

    Args:
        new_vertices: (15, 2) vertices of new tree
        existing_vertices: (n_existing, 15, 2) vertices of placed trees

    Returns:
        True if new tree overlaps with any existing tree
    """
    n_existing = existing_vertices.shape[0]

    for i in range(n_existing):
        if polygons_overlap(new_vertices, existing_vertices[i]):
            return True

    return False


@njit(cache=True, parallel=True)
def transform_trees_batch(configs: np.ndarray) -> np.ndarray:
    """
    Transform multiple trees given their configurations.

    Args:
        configs: (n_trees, 3) - [x, y, angle_deg] per tree

    Returns:
        vertices: (n_trees, 15, 2)
    """
    n_trees = configs.shape[0]
    result = np.empty((n_trees, 15, 2), dtype=np.float64)

    for i in prange(n_trees):
        x = configs[i, 0]
        y = configs[i, 1]
        angle_deg = configs[i, 2]

        angle_rad = angle_deg * (np.pi / 180.0)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Unrolled transform for Numba performance
        result[i, 0, 0] = 0.0 * cos_a - 0.8 * sin_a + x
        result[i, 0, 1] = 0.0 * sin_a + 0.8 * cos_a + y
        result[i, 1, 0] = 0.125 * cos_a - 0.5 * sin_a + x
        result[i, 1, 1] = 0.125 * sin_a + 0.5 * cos_a + y
        result[i, 2, 0] = 0.0625 * cos_a - 0.5 * sin_a + x
        result[i, 2, 1] = 0.0625 * sin_a + 0.5 * cos_a + y
        result[i, 3, 0] = 0.2 * cos_a - 0.25 * sin_a + x
        result[i, 3, 1] = 0.2 * sin_a + 0.25 * cos_a + y
        result[i, 4, 0] = 0.1 * cos_a - 0.25 * sin_a + x
        result[i, 4, 1] = 0.1 * sin_a + 0.25 * cos_a + y
        result[i, 5, 0] = 0.35 * cos_a - 0.0 * sin_a + x
        result[i, 5, 1] = 0.35 * sin_a + 0.0 * cos_a + y
        result[i, 6, 0] = 0.075 * cos_a - 0.0 * sin_a + x
        result[i, 6, 1] = 0.075 * sin_a + 0.0 * cos_a + y
        result[i, 7, 0] = 0.075 * cos_a - (-0.2) * sin_a + x
        result[i, 7, 1] = 0.075 * sin_a + (-0.2) * cos_a + y
        result[i, 8, 0] = -0.075 * cos_a - (-0.2) * sin_a + x
        result[i, 8, 1] = -0.075 * sin_a + (-0.2) * cos_a + y
        result[i, 9, 0] = -0.075 * cos_a - 0.0 * sin_a + x
        result[i, 9, 1] = -0.075 * sin_a + 0.0 * cos_a + y
        result[i, 10, 0] = -0.35 * cos_a - 0.0 * sin_a + x
        result[i, 10, 1] = -0.35 * sin_a + 0.0 * cos_a + y
        result[i, 11, 0] = -0.1 * cos_a - 0.25 * sin_a + x
        result[i, 11, 1] = -0.1 * sin_a + 0.25 * cos_a + y
        result[i, 12, 0] = -0.2 * cos_a - 0.25 * sin_a + x
        result[i, 12, 1] = -0.2 * sin_a + 0.25 * cos_a + y
        result[i, 13, 0] = -0.0625 * cos_a - 0.5 * sin_a + x
        result[i, 13, 1] = -0.0625 * sin_a + 0.5 * cos_a + y
        result[i, 14, 0] = -0.125 * cos_a - 0.5 * sin_a + x
        result[i, 14, 1] = -0.125 * sin_a + 0.5 * cos_a + y

    return result


def benchmark_collision():
    """Benchmark collision detection performance vs expected Rust times."""
    print("Polygon Collision Benchmark (Numba)")
    print("=" * 50)

    # Warmup (JIT compilation)
    print("Warming up JIT compilation...")
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(0.5, 0.0, 45.0)
    for _ in range(100):
        polygons_overlap(poly1, poly2)
        polygons_overlap_fast(poly1, poly2)

    # Warmup batch transform
    warmup_configs = np.zeros((200, 3), dtype=np.float64)
    warmup_configs[:, 0] = np.linspace(-5, 5, 200)
    warmup_configs[:, 2] = 45.0
    warmup_verts = transform_trees_batch(warmup_configs)
    _ = count_overlaps_batch(warmup_verts)

    # Benchmark single pair collision
    print("\n1. Single pair collision:")
    n_iters = 100000

    # Overlapping pair
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(0.3, 0.0, 45.0)  # Close enough to overlap

    start = time.perf_counter()
    for _ in range(n_iters):
        polygons_overlap(poly1, poly2)
    elapsed = time.perf_counter() - start
    print(f"   With safety margin: {elapsed/n_iters*1e6:.3f} µs/call ({n_iters/elapsed:.0f} calls/sec)")

    start = time.perf_counter()
    for _ in range(n_iters):
        polygons_overlap_fast(poly1, poly2)
    elapsed = time.perf_counter() - start
    print(f"   Fast (no margin):   {elapsed/n_iters*1e6:.3f} µs/call ({n_iters/elapsed:.0f} calls/sec)")

    # Non-overlapping pair (bbox filter kicks in)
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(2.0, 0.0, 45.0)  # Far apart

    start = time.perf_counter()
    for _ in range(n_iters):
        polygons_overlap_fast(poly1, poly2)
    elapsed = time.perf_counter() - start
    print(f"   Non-overlap (bbox): {elapsed/n_iters*1e6:.3f} µs/call ({n_iters/elapsed:.0f} calls/sec)")

    # Benchmark batch checking for n=200
    print("\n2. Full configuration check (n=200):")
    n_trees = 200

    # Generate random configuration
    np.random.seed(42)
    configs = np.zeros((n_trees, 3), dtype=np.float64)
    configs[:, 0] = np.random.uniform(-5, 5, n_trees)  # x
    configs[:, 1] = np.random.uniform(-5, 5, n_trees)  # y
    configs[:, 2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315], n_trees)  # angle

    # Transform all trees
    start = time.perf_counter()
    vertices = transform_trees_batch(configs)
    transform_time = time.perf_counter() - start
    print(f"   Transform time: {transform_time*1000:.3f} ms")

    # Count all overlaps
    n_bench = 10
    start = time.perf_counter()
    for _ in range(n_bench):
        n_overlaps = count_overlaps_batch(vertices)
    elapsed = (time.perf_counter() - start) / n_bench
    n_pairs = n_trees * (n_trees - 1) // 2
    print(f"   Check all pairs: {elapsed*1000:.3f} ms ({n_pairs} pairs)")
    print(f"   Per pair: {elapsed/n_pairs*1e6:.3f} µs")
    print(f"   Overlaps found: {n_overlaps}")

    # Benchmark incremental check (adding one tree)
    print("\n3. Incremental check (1 tree vs 199):")
    new_vertices = transform_tree(0.0, 0.0, 0.0)
    existing_vertices = vertices[1:]  # 199 trees

    n_iters = 1000
    start = time.perf_counter()
    for _ in range(n_iters):
        check_tree_overlaps_any(new_vertices, existing_vertices)
    elapsed = time.perf_counter() - start
    print(f"   Time: {elapsed/n_iters*1000:.3f} ms/check")
    print(f"   Per pair: {elapsed/n_iters/199*1e6:.3f} µs")

    # Comparison with expected Rust times
    print("\n4. Comparison with Rust (estimated):")
    rust_pair_time = 0.1  # µs (estimated from previous benchmarks)
    numba_pair_time = elapsed/n_iters/199*1e6
    print(f"   Rust (estimated): ~{rust_pair_time:.1f} µs/pair")
    print(f"   Numba:            {numba_pair_time:.3f} µs/pair")
    print(f"   Ratio: {numba_pair_time/rust_pair_time:.1f}x")


def test_correctness():
    """Test that collision detection matches expected results."""
    print("\nCorrectness Tests")
    print("=" * 50)

    # Test 1: Non-overlapping trees
    print("\n1. Non-overlapping trees (far apart):")
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(2.0, 0.0, 0.0)
    result = polygons_overlap(poly1, poly2)
    expected = False
    print(f"   Result: {result}, Expected: {expected} - {'PASS' if result == expected else 'FAIL'}")

    # Test 2: Clearly overlapping trees
    print("\n2. Clearly overlapping trees:")
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(0.1, 0.0, 0.0)
    result = polygons_overlap(poly1, poly2)
    expected = True
    print(f"   Result: {result}, Expected: {expected} - {'PASS' if result == expected else 'FAIL'}")

    # Test 3: Trees that would pass bbox but not polygon check
    print("\n3. Bbox overlap but polygon OK:")
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(0.6, 0.0, 45.0)  # Rotated, fits in gap
    result_overlap = polygons_overlap(poly1, poly2)

    # Check bbox overlap manually
    min1x, min1y, max1x, max1y = polygon_bounds(poly1)
    min2x, min2y, max2x, max2y = polygon_bounds(poly2)
    bbox_overlap = not (max1x < min2x or max2x < min1x or max1y < min2y or max2y < min1y)

    print(f"   BBox overlap: {bbox_overlap}")
    print(f"   Polygon overlap: {result_overlap}")

    # Test 4: One tree inside another (containment)
    print("\n4. Containment test (small tree inside large area):")
    # This shouldn't happen with our tree shape, but test the logic
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(0.0, 0.3, 0.0)  # Slightly offset
    result = polygons_overlap(poly1, poly2)
    print(f"   Result: {result} (should be True - overlapping)")

    # Test 5: Touching edges (safety margin should catch this)
    print("\n5. Nearly touching (within safety margin):")
    # Trees placed so they just barely don't touch geometrically
    poly1 = transform_tree(0.0, 0.0, 0.0)
    poly2 = transform_tree(0.70001, 0.0, 0.0)  # Just past tree width
    result = polygons_overlap(poly1, poly2)
    print(f"   Distance: ~0.00001 units")
    print(f"   Result: {result} (safety margin = 1e-5)")

    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == '__main__':
    test_correctness()
    print("\n")
    benchmark_collision()
