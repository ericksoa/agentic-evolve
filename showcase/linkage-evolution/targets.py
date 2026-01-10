"""
Target shapes for linkage evolution.

Each target is defined as a list of (x, y) points that form a closed or open path.
The linkage will be evolved to trace these shapes as closely as possible.
"""

import math
from typing import List, Tuple
import numpy as np


def heart_shape(num_points: int = 100, scale: float = 2.0, center: Tuple[float, float] = (2.5, 2.0)) -> List[Tuple[float, float]]:
    """
    Generate a heart-shaped target path.

    Uses parametric heart curve:
    x = 16*sin³(t)
    y = 13*cos(t) - 5*cos(2t) - 2*cos(3t) - cos(4t)
    """
    points = []
    cx, cy = center

    for i in range(num_points):
        t = 2 * math.pi * i / num_points

        # Parametric heart
        x = 16 * (math.sin(t) ** 3)
        y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)

        # Scale and center
        x = x * scale / 16 + cx
        y = y * scale / 16 + cy

        points.append((x, y))

    return points


def figure_eight(num_points: int = 100, scale: float = 1.5, center: Tuple[float, float] = (2.5, 1.5)) -> List[Tuple[float, float]]:
    """
    Generate a figure-8 (lemniscate) target path.

    Uses Bernoulli lemniscate:
    x = a * cos(t) / (1 + sin²(t))
    y = a * sin(t) * cos(t) / (1 + sin²(t))
    """
    points = []
    cx, cy = center

    for i in range(num_points):
        t = 2 * math.pi * i / num_points

        denom = 1 + math.sin(t) ** 2
        x = scale * math.cos(t) / denom
        y = scale * math.sin(t) * math.cos(t) / denom

        points.append((x + cx, y + cy))

    return points


def straight_line(num_points: int = 100, start: Tuple[float, float] = (1.0, 2.0), end: Tuple[float, float] = (4.0, 2.0)) -> List[Tuple[float, float]]:
    """
    Generate a straight line target path.

    This is historically significant - the Peaucellier-Lipkin linkage (1864)
    was the first linkage proven to trace a perfect straight line.
    A 4-bar linkage can only approximate this.
    """
    points = []
    x1, y1 = start
    x2, y2 = end

    for i in range(num_points):
        t = i / (num_points - 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((x, y))

    return points


def circle(num_points: int = 100, radius: float = 1.0, center: Tuple[float, float] = (2.5, 1.5)) -> List[Tuple[float, float]]:
    """
    Generate a circular target path.

    Interestingly, a 4-bar linkage cannot trace a perfect circle,
    but can approximate one reasonably well.
    """
    points = []
    cx, cy = center

    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(t)
        y = cy + radius * math.sin(t)
        points.append((x, y))

    return points


def ellipse(num_points: int = 100, a: float = 2.0, b: float = 1.0, center: Tuple[float, float] = (2.5, 1.5)) -> List[Tuple[float, float]]:
    """
    Generate an elliptical target path.
    """
    points = []
    cx, cy = center

    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        x = cx + a * math.cos(t)
        y = cy + b * math.sin(t)
        points.append((x, y))

    return points


def star(num_points: int = 100, outer_r: float = 2.0, inner_r: float = 0.8,
         num_spikes: int = 5, center: Tuple[float, float] = (2.5, 1.5)) -> List[Tuple[float, float]]:
    """
    Generate a star-shaped target path.
    """
    points = []
    cx, cy = center
    total_vertices = num_spikes * 2

    for i in range(num_points):
        # Which segment are we in?
        t = i / num_points
        vertex_idx = int(t * total_vertices) % total_vertices
        segment_t = (t * total_vertices) % 1.0

        # Alternating between outer and inner radius
        r1 = outer_r if vertex_idx % 2 == 0 else inner_r
        r2 = inner_r if vertex_idx % 2 == 0 else outer_r

        angle1 = 2 * math.pi * vertex_idx / total_vertices - math.pi/2
        angle2 = 2 * math.pi * (vertex_idx + 1) / total_vertices - math.pi/2

        x1, y1 = r1 * math.cos(angle1), r1 * math.sin(angle1)
        x2, y2 = r2 * math.cos(angle2), r2 * math.sin(angle2)

        x = x1 + segment_t * (x2 - x1) + cx
        y = y1 + segment_t * (y2 - y1) + cy
        points.append((x, y))

    return points


def letter_D(num_points: int = 100, scale: float = 2.0, center: Tuple[float, float] = (2.5, 1.5)) -> List[Tuple[float, float]]:
    """
    Generate a letter 'D' shape - good for demonstrating evolution.
    """
    points = []
    cx, cy = center

    # D consists of a vertical line and a semicircle
    # Parameterize: 0-0.33 = bottom to top line, 0.33-1.0 = semicircle back down

    for i in range(num_points):
        t = i / num_points

        if t < 0.25:
            # Bottom to top on left edge
            x = -0.5
            y = -1.0 + t * 4 * 2.0  # Goes from -1 to 1
        elif t < 0.75:
            # Semicircle on right
            angle = math.pi/2 - (t - 0.25) * 2 * math.pi  # From 90° to -90°
            x = -0.5 + 1.0 + 1.0 * math.cos(angle)  # Center of arc at x=0.5
            y = 1.0 * math.sin(angle)
        else:
            # Top to bottom back to start
            x = -0.5
            y = -1.0 + (1.0 - t) * 4 * 2.0

        points.append((x * scale/2 + cx, y * scale/2 + cy))

    return points


# Dictionary of available targets
TARGETS = {
    'heart': heart_shape,
    'figure8': figure_eight,
    'line': straight_line,
    'circle': circle,
    'ellipse': ellipse,
    'star': star,
    'letter_d': letter_D,
}


def get_target(name: str, **kwargs) -> List[Tuple[float, float]]:
    """Get a target shape by name."""
    if name not in TARGETS:
        raise ValueError(f"Unknown target: {name}. Available: {list(TARGETS.keys())}")
    return TARGETS[name](**kwargs)


if __name__ == "__main__":
    # Test all targets
    for name in TARGETS:
        points = get_target(name, num_points=50)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        print(f"{name}: {len(points)} points, X=[{min(xs):.2f}, {max(xs):.2f}], Y=[{min(ys):.2f}, {max(ys):.2f}]")
