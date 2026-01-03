#!/usr/bin/env python3
"""
Analyze the Christmas tree polygon geometry to find efficient packing patterns.
"""

import numpy as np

TREE_VERTICES = np.array([
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
])


def polygon_area(vertices):
    """Calculate polygon area using shoelace formula."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def polygon_bounds(vertices):
    """Get bounding box."""
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    return min_x, min_y, max_x, max_y


def rotate_polygon(vertices, angle_deg):
    """Rotate polygon."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return vertices @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])


def main():
    print("Christmas Tree Geometry Analysis")
    print("=" * 50)

    # Basic properties
    area = polygon_area(TREE_VERTICES)
    bounds = polygon_bounds(TREE_VERTICES)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    bbox_area = width * height

    print(f"\nTree Properties:")
    print(f"  Vertices: {len(TREE_VERTICES)}")
    print(f"  Area: {area:.4f}")
    print(f"  Bounding box: {width:.4f} x {height:.4f} = {bbox_area:.4f}")
    print(f"  Fill ratio (area/bbox): {area/bbox_area:.2%}")
    print(f"  Height: {height:.2f} (from {bounds[1]:.2f} to {bounds[3]:.2f})")
    print(f"  Width: {width:.2f} (from {bounds[0]:.2f} to {bounds[2]:.2f})")

    # Analyze at different rotations
    print(f"\nRotated bounding boxes:")
    for angle in [0, 45, 90, 135]:
        rotated = rotate_polygon(TREE_VERTICES, angle)
        bounds = polygon_bounds(rotated)
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        print(f"  {angle:3d}°: {w:.4f} x {h:.4f} = {w*h:.4f}")

    # Theoretical packing analysis
    print(f"\n" + "=" * 50)
    print("Theoretical Packing Analysis")
    print("=" * 50)

    # For N trees with area A each:
    # - Minimum possible square side: sqrt(N * A)
    # - Actual side depends on packing efficiency
    # - Perfect hexagonal packing achieves ~90.69% efficiency
    # - For irregular shapes, expect 60-80%

    print(f"\nTheoretical minimum box sizes (assuming 100% fill):")
    for n in [1, 5, 10, 25, 50, 100, 200]:
        min_side = np.sqrt(n * area)
        print(f"  N={n:3d}: sqrt({n}*{area:.3f}) = {min_side:.4f}")

    print(f"\nEstimated realistic box sizes (70% fill efficiency):")
    for n in [1, 5, 10, 25, 50, 100, 200]:
        realistic_side = np.sqrt(n * area / 0.70)
        score_contrib = realistic_side**2 / n
        print(f"  N={n:3d}: {realistic_side:.4f}, score contrib: {score_contrib:.4f}")

    # Score analysis
    print(f"\n" + "=" * 50)
    print("Score Analysis")
    print("=" * 50)

    # For optimal packing (side ~ sqrt(n * area / efficiency)):
    # score = sum(side^2 / n) = sum(n * area / efficiency / n) = N * area / efficiency
    # So optimal score ~ 200 * 0.35 / 0.70 = 100 for 70% efficiency
    # Current ~88 suggests about 77% efficiency
    # Target ~70 suggests about 100% efficiency (impossible!) or the formula is wrong

    current_score = 88
    target_score = 70
    n_max = 200

    current_efficiency = n_max * area / current_score
    target_efficiency = n_max * area / target_score

    print(f"\nCurrent score: {current_score}")
    print(f"Target score: {target_score}")
    print(f"Tree area: {area:.4f}")
    print(f"Sum of (area) for 200 trees: {200 * area:.2f}")

    print(f"\nImplied packing efficiency:")
    print(f"  Current: {current_efficiency:.2%}")
    print(f"  Target: {target_efficiency:.2%}")

    # Score breakdown hypothesis
    print(f"\n" + "=" * 50)
    print("Score Breakdown Hypothesis")
    print("=" * 50)

    print("\nIf side = k * sqrt(n), then:")
    print("  score = sum(k^2 * n / n) = sum(k^2) = 200 * k^2")
    print(f"  For score 88: k^2 = 88/200 = 0.44, k = 0.66")
    print(f"  For score 70: k^2 = 70/200 = 0.35, k = 0.59")
    print(f"\nThis means top solutions pack 0.59/0.66 = 11% tighter per dimension")
    print("Or equivalently, 20% better area efficiency.")

    # Interlocking analysis
    print(f"\n" + "=" * 50)
    print("Interlocking Pattern Analysis")
    print("=" * 50)

    print("\nTree shape features for interlocking:")
    print("  - Wide base (0.7 units)")
    print("  - Narrow trunk (0.15 units)")
    print("  - Pointy top")
    print("  - Stepped sides (3 tiers)")

    print("\nPotential interlocking patterns:")
    print("  1. Trunk-in-gap: Place one tree's trunk in another's tier gap")
    print("  2. Tip-in-gap: Nestle tips into base gaps")
    print("  3. Alternating rotations: 0°/180° rows, 45°/225° columns")
    print("  4. Herringbone: 45° offset rows")

    # Key dimensions for interlocking
    print("\nKey dimensions:")
    print(f"  Tier gaps:")
    print(f"    Top tier: y=0.5, width 0.25")
    print(f"    Mid tier: y=0.25, width 0.4")
    print(f"    Base: y=0, width 0.7")
    print(f"  Trunk dimensions: 0.15 wide, 0.2 tall")

    # Calculate minimum spacing between trees
    print("\n" + "=" * 50)
    print("Minimum Spacing Analysis")
    print("=" * 50)

    # For two trees at same rotation, minimum distance depends on direction
    # This is essentially the no-fit polygon radius in each direction

    for angle in [0, 45, 90, 135, 180]:
        # Sample positions in this direction
        rad = np.radians(angle)
        dx, dy = np.cos(rad), np.sin(rad)

        # Find minimum distance where they don't overlap
        # (Approximation using bounding box)
        tree = TREE_VERTICES
        b = polygon_bounds(tree)

        # Project tree extent in direction
        extents = tree @ np.array([dx, dy])
        min_ext = extents.min()
        max_ext = extents.max()
        span = max_ext - min_ext

        print(f"  Direction {angle:3d}°: min distance ~ {span:.4f}")


if __name__ == '__main__':
    main()
