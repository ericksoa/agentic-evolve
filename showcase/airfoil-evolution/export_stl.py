#!/usr/bin/env python3
"""
STL Exporter for Evolved Airfoils

Creates 3D-printable wing sections from airfoil designs.

Output options:
1. Wing section - extruded airfoil for wind tunnel testing
2. Full wing - tapered wing with root and tip profiles
3. Display model - thick base with airfoil cross-section

Usage:
    python export_stl.py airfoil.json output.stl --chord=100 --span=50
"""

import struct
import math
import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from airfoil import Airfoil, load_airfoil


@dataclass
class Triangle:
    """Triangle for STL export."""
    v1: Tuple[float, float, float]
    v2: Tuple[float, float, float]
    v3: Tuple[float, float, float]
    normal: Tuple[float, float, float] = (0, 0, 1)


def compute_normal(v1, v2, v3) -> Tuple[float, float, float]:
    """Compute surface normal from vertices."""
    # Edge vectors
    u = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    v = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])

    # Cross product
    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]

    # Normalize
    length = math.sqrt(nx*nx + ny*ny + nz*nz)
    if length > 0.0001:
        return (nx/length, ny/length, nz/length)
    return (0, 0, 1)


def write_stl_binary(triangles: List[Triangle], filepath: str, name: str = "airfoil"):
    """Write triangles to binary STL file."""
    with open(filepath, 'wb') as f:
        # Header
        header = f"Binary STL - {name}".encode('ascii')[:80].ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))

        # Triangles
        for tri in triangles:
            normal = compute_normal(tri.v1, tri.v2, tri.v3)
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *tri.v1))
            f.write(struct.pack('<3f', *tri.v2))
            f.write(struct.pack('<3f', *tri.v3))
            f.write(struct.pack('<H', 0))  # Attribute

    print(f"Wrote {len(triangles)} triangles to {filepath}")


def create_wing_section(airfoil: Airfoil,
                        chord: float = 100.0,  # mm
                        span: float = 50.0,    # mm
                        n_points: int = 80) -> List[Triangle]:
    """
    Create a rectangular wing section (extruded airfoil).

    The airfoil is extruded along the Z-axis (span direction).
    - X: chord direction (leading edge at 0, trailing edge at chord)
    - Y: thickness direction
    - Z: span direction
    """
    triangles = []

    # Get airfoil coordinates
    x_coords, y_coords = airfoil.get_coordinates(n_points)

    # Scale to chord length
    x_scaled = [x * chord for x in x_coords]
    y_scaled = [y * chord for y in y_coords]

    n = len(x_scaled)

    # Create side faces (extruded profile)
    z0, z1 = 0.0, span

    for i in range(n):
        i_next = (i + 1) % n

        # Four corners of this quad
        p1 = (x_scaled[i], y_scaled[i], z0)
        p2 = (x_scaled[i_next], y_scaled[i_next], z0)
        p3 = (x_scaled[i_next], y_scaled[i_next], z1)
        p4 = (x_scaled[i], y_scaled[i], z1)

        # Two triangles per quad
        triangles.append(Triangle(p1, p2, p3))
        triangles.append(Triangle(p1, p3, p4))

    # Create end caps
    # Find center point
    cx = sum(x_scaled) / n
    cy = sum(y_scaled) / n

    # Bottom cap (z=0)
    for i in range(n):
        i_next = (i + 1) % n
        p1 = (cx, cy, z0)
        p2 = (x_scaled[i_next], y_scaled[i_next], z0)
        p3 = (x_scaled[i], y_scaled[i], z0)
        triangles.append(Triangle(p1, p2, p3))

    # Top cap (z=span)
    for i in range(n):
        i_next = (i + 1) % n
        p1 = (cx, cy, z1)
        p2 = (x_scaled[i], y_scaled[i], z1)
        p3 = (x_scaled[i_next], y_scaled[i_next], z1)
        triangles.append(Triangle(p1, p2, p3))

    return triangles


def create_display_stand(airfoil: Airfoil,
                         chord: float = 100.0,
                         base_height: float = 10.0,
                         label_height: float = 3.0) -> List[Triangle]:
    """
    Create a wing section on a display stand with flat base.
    Good for desk display or wind tunnel mounting.
    """
    triangles = []

    # Wing section on top
    wing = create_wing_section(airfoil, chord=chord, span=chord*0.3)

    # Offset wing upward
    for tri in wing:
        v1 = (tri.v1[0], tri.v1[1] + base_height, tri.v1[2])
        v2 = (tri.v2[0], tri.v2[1] + base_height, tri.v2[2])
        v3 = (tri.v3[0], tri.v3[1] + base_height, tri.v3[2])
        triangles.append(Triangle(v1, v2, v3))

    # Base plate
    base_length = chord * 1.2
    base_width = chord * 0.5
    margin = chord * 0.1

    # Base corners
    bx0, bx1 = -margin, chord + margin
    bz0, bz1 = -margin, chord * 0.3 + margin

    # Bottom face
    triangles.append(Triangle(
        (bx0, 0, bz0), (bx1, 0, bz1), (bx1, 0, bz0)
    ))
    triangles.append(Triangle(
        (bx0, 0, bz0), (bx0, 0, bz1), (bx1, 0, bz1)
    ))

    # Top face (at base_height, except where wing is)
    triangles.append(Triangle(
        (bx0, base_height, bz0), (bx1, base_height, bz0), (bx1, base_height, bz1)
    ))
    triangles.append(Triangle(
        (bx0, base_height, bz0), (bx1, base_height, bz1), (bx0, base_height, bz1)
    ))

    # Side faces of base
    for (x0, z0), (x1, z1) in [
        ((bx0, bz0), (bx1, bz0)),  # Front
        ((bx1, bz0), (bx1, bz1)),  # Right
        ((bx1, bz1), (bx0, bz1)),  # Back
        ((bx0, bz1), (bx0, bz0)),  # Left
    ]:
        triangles.append(Triangle(
            (x0, 0, z0), (x1, 0, z1), (x1, base_height, z1)
        ))
        triangles.append(Triangle(
            (x0, 0, z0), (x1, base_height, z1), (x0, base_height, z0)
        ))

    return triangles


def export_airfoil_stl(airfoil: Airfoil,
                       output_path: str,
                       chord: float = 100.0,
                       span: float = 50.0,
                       style: str = "section"):
    """
    Export airfoil as STL file.

    Args:
        airfoil: Airfoil object to export
        output_path: Output STL file path
        chord: Chord length in mm
        span: Span length in mm (for section style)
        style: "section" for plain wing, "display" for display stand
    """
    if style == "display":
        triangles = create_display_stand(airfoil, chord=chord)
    else:
        triangles = create_wing_section(airfoil, chord=chord, span=span)

    write_stl_binary(triangles, output_path, "evolved_airfoil")


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python export_stl.py <airfoil.json> <output.stl> [options]")
        print("\nOptions:")
        print("  --chord=N     Chord length in mm (default: 100)")
        print("  --span=N      Span length in mm (default: 50)")
        print("  --style=X     'section' or 'display' (default: section)")
        sys.exit(1)

    airfoil_path = sys.argv[1]
    output_path = sys.argv[2]

    chord = 100.0
    span = 50.0
    style = "section"

    for arg in sys.argv[3:]:
        if arg.startswith("--chord="):
            chord = float(arg.split("=")[1])
        elif arg.startswith("--span="):
            span = float(arg.split("=")[1])
        elif arg.startswith("--style="):
            style = arg.split("=")[1]

    try:
        airfoil = load_airfoil(airfoil_path)
    except Exception as e:
        print(f"Error loading airfoil: {e}")
        sys.exit(1)

    export_airfoil_stl(airfoil, output_path, chord=chord, span=span, style=style)

    print(f"\nExported {style} wing:")
    print(f"  Chord: {chord} mm")
    if style == "section":
        print(f"  Span: {span} mm")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
