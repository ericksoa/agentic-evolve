#!/usr/bin/env python3
"""Quick optimization for n=2,6,7 with strict validation."""

import numpy as np
import math
import json
import csv

TREE_VERTICES = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

def get_vertices(x, y, angle):
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return [(vx * cos_a - vy * sin_a + x, vx * sin_a + vy * cos_a + y) for vx, vy in TREE_VERTICES]

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    d1, d2 = ccw(A, B, C), ccw(A, B, D)
    d3, d4 = ccw(C, D, A), ccw(C, D, B)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
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

def has_overlap(trees):
    polys = [get_vertices(x, y, a) for x, y, a in trees]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(polys[i], polys[j]):
                return True
    return False

def compute_side(trees):
    all_verts = []
    for x, y, a in trees:
        all_verts.extend(get_vertices(x, y, a))
    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    return max(max(xs) - min(xs), max(ys) - min(ys))

def sa_optimize(trees, iterations=50000):
    n = len(trees)
    trees = list(trees)

    # If overlapping, spread out first
    scale = 1.0
    while has_overlap(trees):
        scale *= 1.05
        trees = [(x * scale, y * scale, a) for x, y, a in trees]
        if scale > 3.0:
            return float('inf'), None

    best_trees = list(trees)
    current_side = compute_side(trees)
    best_side = current_side

    T = 0.2
    Tf = 0.0001
    alpha = (Tf / T) ** (1 / iterations)

    for _ in range(iterations):
        idx = np.random.randint(0, n)
        x, y, a = trees[idx]

        sc = max(0.01, T * 2)
        dx = np.random.normal(0, sc * 0.1)
        dy = np.random.normal(0, sc * 0.1)
        da = np.random.normal(0, sc * 10)

        new_tree = (x + dx, y + dy, (a + da) % 360)
        candidate = trees[:idx] + [new_tree] + trees[idx + 1:]

        if has_overlap(candidate):
            T *= alpha
            continue

        new_side = compute_side(candidate)
        delta = new_side - current_side

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            trees = candidate
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_trees = list(trees)

        T *= alpha

    return best_side, best_trees

def main():
    # Load current submission
    with open('submission_best.csv') as f:
        reader = csv.DictReader(f)
        groups = {}
        for row in reader:
            n = int(row['id'].split('_')[0])
            x = float(row['x'].lstrip('s'))
            y = float(row['y'].lstrip('s'))
            deg = float(row['deg'].lstrip('s'))
            if n not in groups:
                groups[n] = []
            groups[n].append((x, y, deg))

    results = {}

    # n=2: Try 180-degree opposed pattern
    print("n=2:")
    current_side = compute_side(groups[2])
    print(f"  Current: {current_side:.6f}")

    best_side = current_side
    best_trees = groups[2]

    # Try different angle pairs
    for a1 in range(0, 180, 15):
        a2 = a1 + 180
        for d in [0.3, 0.4, 0.5, 0.6]:
            init = [(0, 0, a1), (d, d, a2)]
            side, trees = sa_optimize(init, 30000)
            if trees and side < best_side and not has_overlap(trees):
                best_side = side
                best_trees = trees
                print(f"  New best: {side:.6f} (a={a1},{a2}, d={d})")

    if best_side < current_side:
        results['2'] = {'side': best_side, 'trees': best_trees}
        print(f"  Final: {best_side:.6f} (improved!)")
    else:
        print(f"  No improvement")

    # n=6
    print("\nn=6:")
    current_side = compute_side(groups[6])
    print(f"  Current: {current_side:.6f}")

    best_side = current_side
    best_trees = groups[6]

    # Try circular patterns
    for r in [0.4, 0.5, 0.6]:
        for ba in [0, 30, 45, 60]:
            init = [(r * math.cos(2*math.pi*i/6), r * math.sin(2*math.pi*i/6), (ba + i*60) % 360)
                    for i in range(6)]
            side, trees = sa_optimize(init, 40000)
            if trees and side < best_side and not has_overlap(trees):
                best_side = side
                best_trees = trees
                print(f"  New best: {side:.6f}")

    if best_side < current_side:
        results['6'] = {'side': best_side, 'trees': best_trees}
        print(f"  Final: {best_side:.6f} (improved!)")
    else:
        print(f"  No improvement")

    # n=7
    print("\nn=7:")
    current_side = compute_side(groups[7])
    print(f"  Current: {current_side:.6f}")

    best_side = current_side
    best_trees = groups[7]

    # Try circular with center
    for r in [0.5, 0.6, 0.7]:
        for ba in [0, 30, 45]:
            init = [(0, 0, ba)]  # center
            init += [(r * math.cos(2*math.pi*i/6), r * math.sin(2*math.pi*i/6), (ba + 60 + i*60) % 360)
                     for i in range(6)]
            side, trees = sa_optimize(init, 40000)
            if trees and side < best_side and not has_overlap(trees):
                best_side = side
                best_trees = trees
                print(f"  New best: {side:.6f}")

    if best_side < current_side:
        results['7'] = {'side': best_side, 'trees': best_trees}
        print(f"  Final: {best_side:.6f} (improved!)")
    else:
        print(f"  No improvement")

    # Save results
    if results:
        with open('python/gen113_fixed.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} improvements to gen113_fixed.json")
    else:
        print("\nNo improvements found")

if __name__ == '__main__':
    main()
