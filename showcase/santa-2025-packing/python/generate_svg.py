#!/usr/bin/env python3
"""Generate SVG visualizations from submission CSV."""

import csv
import math
import sys
from pathlib import Path

# Tree vertices (same as Rust code)
TREE_VERTICES = [
    (0.0, 0.8),      # Tip
    (0.125, 0.5),    # top_w/2
    (0.0625, 0.5),   # top_w/4
    (0.2, 0.25),     # mid_w/2
    (0.1, 0.25),     # mid_w/4
    (0.35, 0.0),     # base_w/2
    (0.075, 0.0),    # trunk_w/2
    (0.075, -0.2),   # trunk_w/2, trunk_bottom
    (-0.075, -0.2),  # -trunk_w/2, trunk_bottom
    (-0.075, 0.0),   # -trunk_w/2
    (-0.35, 0.0),    # -base_w/2
    (-0.1, 0.25),    # -mid_w/4
    (-0.2, 0.25),    # -mid_w/2
    (-0.0625, 0.5),  # -top_w/4
    (-0.125, 0.5),   # -top_w/2
]


def rotate_point(x, y, angle_deg):
    """Rotate a point around the origin."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def get_tree_vertices(cx, cy, angle_deg):
    """Get the rotated and translated tree vertices."""
    vertices = []
    for vx, vy in TREE_VERTICES:
        rx, ry = rotate_point(vx, vy, angle_deg)
        vertices.append((rx + cx, ry + cy))
    return vertices


def parse_submission(csv_path, n_filter=None):
    """Parse submission CSV and extract trees for specific n values."""
    trees_by_n = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse id like "004_1" -> n=4, idx=1
            id_parts = row['id'].split('_')
            n = int(id_parts[0])

            if n_filter is not None and n != n_filter:
                continue

            # Parse s-prefixed values
            x = float(row['x'].lstrip('s'))
            y = float(row['y'].lstrip('s'))
            deg = float(row['deg'].lstrip('s'))

            if n not in trees_by_n:
                trees_by_n[n] = []
            trees_by_n[n].append((x, y, deg))

    return trees_by_n


def generate_svg(trees, width=600, height=600):
    """Generate SVG for a list of trees."""
    if not trees:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle">No trees</text></svg>'

    # Calculate bounds
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    all_vertices = []
    for cx, cy, angle_deg in trees:
        vertices = get_tree_vertices(cx, cy, angle_deg)
        all_vertices.append(vertices)
        for x, y in vertices:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    data_width = max_x - min_x
    data_height = max_y - min_y
    side = max(data_width, data_height)

    # Add margin
    margin = side * 0.05
    view_min_x = min_x - margin
    view_min_y = min_y - margin
    view_size = side + 2.0 * margin

    # SVG header with flipped Y axis (SVG Y increases downward)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{view_min_x} {-view_min_y - view_size} {view_size} {view_size}">
  <rect x="{view_min_x}" y="{-view_min_y - view_size}" width="{view_size}" height="{view_size}" fill="#f8f8f8" stroke="#ccc" stroke-width="0.01"/>
  <rect x="{min_x}" y="{-max_y}" width="{side}" height="{side}" fill="none" stroke="#2196F3" stroke-width="0.02" stroke-dasharray="0.05,0.05"/>
'''

    # Draw trees with gradient colors
    n = len(trees)
    for i, vertices in enumerate(all_vertices):
        hue = int(i / n * 120)  # Green gradient
        color = f"hsl({hue}, 70%, 40%)"
        stroke_color = f"hsl({hue}, 70%, 30%)"

        points = " ".join(f"{x:.4f},{-y:.4f}" for x, y in vertices)
        svg += f'  <polygon points="{points}" fill="{color}" stroke="{stroke_color}" stroke-width="0.005" opacity="0.8"/>\n'

    # Add info text
    svg += f'  <text x="{view_min_x + 0.05}" y="{-view_min_y - view_size + 0.2}" font-size="0.15" fill="#333">n={n}, side={side:.4f}</text>\n</svg>'

    return svg


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_svg.py <submission.csv> [n_values...]")
        print("       generate_svg.py submission_best.csv 4 5 10 50 100 150 200")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Default n values to generate
    if len(sys.argv) > 2:
        n_values = [int(x) for x in sys.argv[2:]]
    else:
        n_values = [4, 5, 10, 50, 100, 150, 200]

    print(f"Reading submission from {csv_path}")

    # Parse all trees
    trees_by_n = parse_submission(csv_path)

    # Generate SVGs
    output_dir = Path(csv_path).parent
    for n in n_values:
        if n not in trees_by_n:
            print(f"  n={n}: not found in submission")
            continue

        trees = trees_by_n[n]
        if len(trees) != n:
            print(f"  n={n}: expected {n} trees, got {len(trees)}")
            continue

        # Larger SVG for larger n
        if n <= 10:
            size = 500
        elif n <= 50:
            size = 600
        elif n <= 100:
            size = 700
        else:
            size = 800

        svg = generate_svg(trees, size, size)
        output_path = output_dir / f"packing_n{n}.svg"

        with open(output_path, 'w') as f:
            f.write(svg)

        # Calculate side length from bounds
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        for cx, cy, angle_deg in trees:
            for vx, vy in get_tree_vertices(cx, cy, angle_deg):
                min_x = min(min_x, vx)
                min_y = min(min_y, vy)
                max_x = max(max_x, vx)
                max_y = max(max_y, vy)
        side = max(max_x - min_x, max_y - min_y)

        print(f"  n={n}: side={side:.4f} -> {output_path}")


if __name__ == "__main__":
    main()
