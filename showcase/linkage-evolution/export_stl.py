#!/usr/bin/env python3
"""
STL Exporter for 4-Bar Linkages

Generates 3D-printable STL files for the evolved linkage mechanism.
Creates separate parts for each link with pivot holes.

Output:
- ground_plate.stl: Base plate with fixed pivot mounts
- crank.stl: Input link (attach to motor or handle)
- coupler.stl: Connecting link with tracer point marker
- follower.stl: Output rocker link
- assembly_guide.txt: Assembly instructions
"""

import math
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from linkage import Linkage4Bar, load_linkage


# Design parameters for 3D printing
LINK_WIDTH = 10.0        # Width of link bars (mm)
LINK_THICKNESS = 5.0     # Thickness of links (mm)
PIVOT_HOLE_RADIUS = 2.5  # Radius of pivot holes (for M5 bolts)
PIVOT_BOSS_RADIUS = 6.0  # Radius of boss around pivot holes
SCALE_FACTOR = 20.0      # Scale linkage units to mm


@dataclass
class Triangle:
    """A triangle defined by three vertices and a normal."""
    v1: Tuple[float, float, float]
    v2: Tuple[float, float, float]
    v3: Tuple[float, float, float]
    normal: Tuple[float, float, float] = (0, 0, 1)


def write_stl_binary(triangles: List[Triangle], filepath: str, name: str = "linkage"):
    """Write triangles to binary STL file."""
    with open(filepath, 'wb') as f:
        # Header (80 bytes)
        header = f"Binary STL - {name}".encode('ascii')
        header = header[:80].ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))

        # Triangles
        for tri in triangles:
            # Normal
            f.write(struct.pack('<3f', *tri.normal))
            # Vertices
            f.write(struct.pack('<3f', *tri.v1))
            f.write(struct.pack('<3f', *tri.v2))
            f.write(struct.pack('<3f', *tri.v3))
            # Attribute byte count
            f.write(struct.pack('<H', 0))

    print(f"Wrote {len(triangles)} triangles to {filepath}")


def create_cylinder(center: Tuple[float, float], radius: float, height: float,
                    segments: int = 32, z_base: float = 0) -> List[Triangle]:
    """Create a cylinder as triangles."""
    triangles = []
    cx, cy = center

    for i in range(segments):
        angle1 = 2 * math.pi * i / segments
        angle2 = 2 * math.pi * (i + 1) / segments

        x1, y1 = cx + radius * math.cos(angle1), cy + radius * math.sin(angle1)
        x2, y2 = cx + radius * math.cos(angle2), cy + radius * math.sin(angle2)

        # Bottom face
        triangles.append(Triangle(
            (cx, cy, z_base),
            (x2, y2, z_base),
            (x1, y1, z_base),
            (0, 0, -1)
        ))

        # Top face
        triangles.append(Triangle(
            (cx, cy, z_base + height),
            (x1, y1, z_base + height),
            (x2, y2, z_base + height),
            (0, 0, 1)
        ))

        # Side faces
        nx = math.cos((angle1 + angle2) / 2)
        ny = math.sin((angle1 + angle2) / 2)

        triangles.append(Triangle(
            (x1, y1, z_base),
            (x2, y2, z_base),
            (x1, y1, z_base + height),
            (nx, ny, 0)
        ))
        triangles.append(Triangle(
            (x2, y2, z_base),
            (x2, y2, z_base + height),
            (x1, y1, z_base + height),
            (nx, ny, 0)
        ))

    return triangles


def create_link_bar(start: Tuple[float, float], end: Tuple[float, float],
                    width: float, thickness: float, z_base: float = 0) -> List[Triangle]:
    """Create a rectangular bar link between two points."""
    triangles = []

    sx, sy = start
    ex, ey = end

    # Direction and perpendicular
    dx, dy = ex - sx, ey - sy
    length = math.sqrt(dx*dx + dy*dy)
    if length < 0.001:
        return triangles

    dx, dy = dx/length, dy/length
    px, py = -dy, dx  # Perpendicular

    hw = width / 2  # Half width

    # Four corners of the bar (bottom)
    corners_bottom = [
        (sx + px*hw, sy + py*hw, z_base),
        (sx - px*hw, sy - py*hw, z_base),
        (ex - px*hw, ey - py*hw, z_base),
        (ex + px*hw, ey + py*hw, z_base),
    ]

    # Four corners (top)
    corners_top = [(x, y, z_base + thickness) for x, y, z in corners_bottom]

    # Bottom face
    triangles.append(Triangle(corners_bottom[0], corners_bottom[2], corners_bottom[1], (0, 0, -1)))
    triangles.append(Triangle(corners_bottom[0], corners_bottom[3], corners_bottom[2], (0, 0, -1)))

    # Top face
    triangles.append(Triangle(corners_top[0], corners_top[1], corners_top[2], (0, 0, 1)))
    triangles.append(Triangle(corners_top[0], corners_top[2], corners_top[3], (0, 0, 1)))

    # Side faces
    for i in range(4):
        j = (i + 1) % 4
        b1, b2 = corners_bottom[i], corners_bottom[j]
        t1, t2 = corners_top[i], corners_top[j]

        # Compute normal
        edge = (b2[0]-b1[0], b2[1]-b1[1], 0)
        up = (0, 0, 1)
        normal = (edge[1]*up[2] - edge[2]*up[1],
                  edge[2]*up[0] - edge[0]*up[2],
                  edge[0]*up[1] - edge[1]*up[0])
        nl = math.sqrt(sum(n*n for n in normal))
        if nl > 0.001:
            normal = tuple(n/nl for n in normal)

        triangles.append(Triangle(b1, b2, t1, normal))
        triangles.append(Triangle(b2, t2, t1, normal))

    return triangles


def create_link_with_holes(start: Tuple[float, float], end: Tuple[float, float],
                           width: float = LINK_WIDTH, thickness: float = LINK_THICKNESS,
                           hole_radius: float = PIVOT_HOLE_RADIUS,
                           boss_radius: float = PIVOT_BOSS_RADIUS) -> List[Triangle]:
    """
    Create a link with circular bosses at each end for pivot holes.
    Note: This creates a simplified solid - actual holes would need boolean operations.
    For 3D printing, holes can be drilled after printing or designed in CAD software.
    """
    triangles = []

    # Main bar
    triangles.extend(create_link_bar(start, end, width, thickness))

    # Circular bosses at each end
    triangles.extend(create_cylinder(start, boss_radius, thickness, segments=24))
    triangles.extend(create_cylinder(end, boss_radius, thickness, segments=24))

    return triangles


def export_linkage_stl(linkage: Linkage4Bar, output_dir: str, scale: float = SCALE_FACTOR):
    """
    Export all linkage parts as separate STL files.

    Args:
        linkage: The 4-bar linkage to export
        output_dir: Directory to save STL files
        scale: Scale factor from linkage units to mm
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Scale all dimensions
    L1 = linkage.L1 * scale
    L2 = linkage.L2 * scale
    L3 = linkage.L3 * scale
    L4 = linkage.L4 * scale
    Px = linkage.Px * scale
    Py = linkage.Py * scale

    # Fixed pivot positions
    pivot_A = (0, 0)
    pivot_D = (L1, 0)

    # 1. Crank (L2)
    crank_tris = create_link_with_holes((0, 0), (L2, 0))
    write_stl_binary(crank_tris, str(output_path / "crank.stl"), "crank")

    # 2. Follower (L4)
    follower_tris = create_link_with_holes((0, 0), (L4, 0))
    write_stl_binary(follower_tris, str(output_path / "follower.stl"), "follower")

    # 3. Coupler (L3) with tracer point marker
    coupler_tris = create_link_with_holes((0, 0), (L3, 0))
    # Add tracer point marker (small cylinder)
    tracer_marker = create_cylinder((Px, Py), 3.0, LINK_THICKNESS + 2, segments=16)
    coupler_tris.extend(tracer_marker)
    write_stl_binary(coupler_tris, str(output_path / "coupler.stl"), "coupler")

    # 4. Ground plate
    ground_tris = []
    # Base plate
    plate_margin = 20
    plate_width = L1 + 2 * plate_margin
    plate_height = max(L2 + L3, L4) + 2 * plate_margin
    plate_thickness = 3.0

    # Simple rectangular plate
    ground_tris.extend(create_link_bar(
        (-plate_margin, -plate_height/2),
        (L1 + plate_margin, -plate_height/2),
        plate_height,
        plate_thickness
    ))
    # Pivot mount cylinders
    ground_tris.extend(create_cylinder(pivot_A, PIVOT_BOSS_RADIUS * 1.5, 10, z_base=plate_thickness))
    ground_tris.extend(create_cylinder(pivot_D, PIVOT_BOSS_RADIUS * 1.5, 10, z_base=plate_thickness))
    write_stl_binary(ground_tris, str(output_path / "ground_plate.stl"), "ground")

    # 5. Assembly guide
    guide_path = output_path / "assembly_guide.txt"
    with open(guide_path, 'w') as f:
        f.write("4-BAR LINKAGE ASSEMBLY GUIDE\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Evolved Linkage Parameters:\n")
        f.write(f"  L1 (Ground):   {linkage.L1:.3f} units = {L1:.1f} mm\n")
        f.write(f"  L2 (Crank):    {linkage.L2:.3f} units = {L2:.1f} mm\n")
        f.write(f"  L3 (Coupler):  {linkage.L3:.3f} units = {L3:.1f} mm\n")
        f.write(f"  L4 (Follower): {linkage.L4:.3f} units = {L4:.1f} mm\n")
        f.write(f"  Tracer Point:  ({linkage.Px:.3f}, {linkage.Py:.3f}) on coupler\n\n")
        f.write("Parts List:\n")
        f.write("  - ground_plate.stl (1x) - Base with fixed pivot mounts\n")
        f.write("  - crank.stl (1x) - Input link, attach to pivot A\n")
        f.write("  - coupler.stl (1x) - With tracer point marker\n")
        f.write("  - follower.stl (1x) - Attach to pivot D\n\n")
        f.write("Hardware Needed:\n")
        f.write("  - 4x M5 bolts (for pivots)\n")
        f.write("  - 4x M5 nuts\n")
        f.write("  - 8x M5 washers\n")
        f.write("  - Optional: bearings for smoother motion\n\n")
        f.write("Assembly:\n")
        f.write("  1. Mount ground plate to a stable surface\n")
        f.write("  2. Attach crank to pivot A (left fixed pivot)\n")
        f.write("  3. Attach follower to pivot D (right fixed pivot)\n")
        f.write("  4. Connect coupler between crank and follower\n")
        f.write("  5. The tracer point marker shows where to attach a pen\n\n")
        f.write("Operation:\n")
        f.write("  - Rotate the crank (L2) to trace the evolved curve\n")
        f.write("  - Attach a pen to the tracer point for drawing\n")

    print(f"\nExported linkage to {output_path}/")
    print(f"  - crank.stl")
    print(f"  - coupler.stl")
    print(f"  - follower.stl")
    print(f"  - ground_plate.stl")
    print(f"  - assembly_guide.txt")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python export_stl.py <linkage.py> [output_dir]")
        print("Exports 3D-printable STL files for the linkage mechanism.")
        sys.exit(1)

    linkage_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "stl_output"

    try:
        linkage = load_linkage(linkage_path)
    except Exception as e:
        print(f"Error loading linkage: {e}")
        sys.exit(1)

    export_linkage_stl(linkage, output_dir)


if __name__ == "__main__":
    main()
