#!/usr/bin/env python3
"""
Generate images for the README showcase.
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

from linkage import Linkage4Bar, load_linkage
from targets import get_target


def plot_linkage_diagram():
    """Create a diagram explaining 4-bar linkage components."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Example linkage for visualization
    L1, L2, L3, L4 = 4.0, 1.5, 3.5, 3.0
    theta = math.radians(45)

    # Fixed pivots
    Ax, Ay = 0, 0
    Dx, Dy = L1, 0

    # Moving joints (at theta=45°)
    Bx = Ax + L2 * math.cos(theta)
    By = Ay + L2 * math.sin(theta)

    # Solve for C
    BD = math.sqrt((Bx - Dx)**2 + (By - Dy)**2)
    cos_angle = (BD**2 + L3**2 - L4**2) / (2 * BD * L3)
    cos_angle = max(-1, min(1, cos_angle))
    angle_DBC = math.acos(cos_angle)
    angle_BD = math.atan2(Dy - By, Dx - Bx)
    angle_BC = angle_BD + angle_DBC
    Cx = Bx + L3 * math.cos(angle_BC)
    Cy = By + L3 * math.sin(angle_BC)

    # Tracer point
    Px, Py = 1.75, 1.0
    coupler_len = math.sqrt((Cx - Bx)**2 + (Cy - By)**2)
    ux = (Cx - Bx) / coupler_len
    uy = (Cy - By) / coupler_len
    vx, vy = -uy, ux
    Tx = Bx + Px * ux + Py * vx
    Ty = By + Px * uy + Py * vy

    # Draw ground
    ax.fill([-0.5, L1 + 0.5, L1 + 0.5, -0.5], [-0.3, -0.3, -0.5, -0.5],
            color='#8B4513', alpha=0.5)
    ax.plot([-0.5, L1 + 0.5], [0, 0], 'k-', linewidth=4)

    # Draw fixed pivots
    ax.plot(Ax, Ay, 'ko', markersize=20, markerfacecolor='#444444', zorder=10)
    ax.plot(Dx, Dy, 'ko', markersize=20, markerfacecolor='#444444', zorder=10)

    # Draw links with colors
    ax.plot([Ax, Bx], [Ay, By], color='#E74C3C', linewidth=8, solid_capstyle='round', label='Crank (L2)')
    ax.plot([Bx, Cx], [By, Cy], color='#27AE60', linewidth=8, solid_capstyle='round', label='Coupler (L3)')
    ax.plot([Dx, Cx], [Dy, Cy], color='#3498DB', linewidth=8, solid_capstyle='round', label='Follower (L4)')

    # Draw moving joints
    ax.plot(Bx, By, 'o', markersize=14, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2, zorder=11)
    ax.plot(Cx, Cy, 'o', markersize=14, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2, zorder=11)

    # Draw tracer point
    ax.plot(Tx, Ty, 'o', markersize=16, markerfacecolor='#9B59B6', markeredgecolor='white', markeredgewidth=2, zorder=12)
    ax.plot([Bx, Tx], [By, Ty], '--', color='#9B59B6', linewidth=2, alpha=0.7)

    # Labels
    ax.annotate('Fixed Pivot A', (Ax, Ay), xytext=(Ax - 0.8, Ay - 0.8), fontsize=11, ha='center')
    ax.annotate('Fixed Pivot D', (Dx, Dy), xytext=(Dx + 0.8, Dy - 0.8), fontsize=11, ha='center')
    ax.annotate('Joint B', (Bx, By), xytext=(Bx - 0.5, By + 0.4), fontsize=11)
    ax.annotate('Joint C', (Cx, Cy), xytext=(Cx + 0.3, Cy + 0.3), fontsize=11)
    ax.annotate('Tracer\nPoint', (Tx, Ty), xytext=(Tx + 0.5, Ty + 0.3), fontsize=11,
                color='#9B59B6', fontweight='bold')

    # Ground label
    ax.annotate('Ground (L1)', (L1/2, -0.7), fontsize=11, ha='center')

    # Dimension lines
    ax.annotate('', xy=(L1, -0.15), xytext=(0, -0.15),
                arrowprops=dict(arrowstyle='<->', color='gray'))

    ax.set_xlim(-1.5, L1 + 1.5)
    ax.set_ylim(-1.2, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title('4-Bar Linkage Mechanism', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('images/linkage_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: images/linkage_diagram.png")


def plot_evolution_comparison():
    """Create side-by-side comparison of baseline vs evolved."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Load linkages
    baseline = load_linkage('baseline.py')
    evolved = load_linkage('evolved_figure8.py')

    # Get target
    target = get_target('figure8', num_points=200)
    target_x = [p[0] for p in target]
    target_y = [p[1] for p in target]

    for ax, linkage, title, fitness in [
        (axes[0], baseline, 'Baseline Linkage', 15.08),
        (axes[1], evolved, 'Evolved Linkage', 18.87)
    ]:
        # Trace path
        traced = linkage.trace_path(num_samples=360)
        traced_x = [p[0] for p in traced]
        traced_y = [p[1] for p in traced]

        # Normalize for comparison
        def normalize(xs, ys):
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            scale = max(max(abs(x - cx) for x in xs), max(abs(y - cy) for y in ys))
            return [(x - cx) / scale for x in xs], [(y - cy) / scale for y in ys]

        traced_x, traced_y = normalize(traced_x, traced_y)
        norm_target_x, norm_target_y = normalize(target_x.copy(), target_y.copy())

        # Plot
        ax.plot(norm_target_x, norm_target_y, 'r--', linewidth=2, label='Target (Figure-8)', alpha=0.7)
        ax.plot(traced_x, traced_y, 'b-', linewidth=2.5, label='Traced Path')
        ax.fill(traced_x, traced_y, alpha=0.1, color='blue')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'{title}\nFitness: {fitness:.2f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Y (normalized)')

    plt.suptitle('Evolution Result: +25% Fitness Improvement', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('images/evolution_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: images/evolution_comparison.png")


def plot_fitness_progression():
    """Create fitness progression chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Data from evolution run
    generations = list(range(1, 16))
    fitness = [15.74, 16.76, 17.12, 17.12, 17.23, 17.55, 17.85, 17.96,
               18.03, 18.18, 18.37, 18.67, 18.69, 18.87, 18.87]

    # Baseline
    baseline = 15.08

    # Plot
    ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline})')
    ax.plot(generations, fitness, 'b-o', linewidth=2, markersize=8, label='Evolution Progress')
    ax.fill_between(generations, baseline, fitness, alpha=0.2, color='green')

    # Annotations
    ax.annotate(f'+25%\nimprovement!',
                xy=(15, fitness[-1]), xytext=(12, 19.5),
                fontsize=12, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness Score', fontsize=12)
    ax.set_title('Fitness Progression Over 15 Generations', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)
    ax.set_ylim(14, 20)

    plt.tight_layout()
    plt.savefig('images/fitness_progression.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: images/fitness_progression.png")


def plot_3d_assembly():
    """Create a 3D-style assembly visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Load evolved linkage
    evolved = load_linkage('evolved_figure8.py')
    scale = 20  # mm per unit

    L1 = evolved.L1 * scale
    L2 = evolved.L2 * scale
    L3 = evolved.L3 * scale
    L4 = evolved.L4 * scale

    # Draw isometric-style 3D representation
    offset_3d = 3  # 3D offset for thickness effect

    # Ground plate (gray)
    plate_x = [-20, L1 + 20, L1 + 20, -20]
    plate_y = [-30, -30, 50, 50]
    ax.fill(plate_x, plate_y, color='#CCCCCC', edgecolor='#888888', linewidth=2)
    ax.fill([x + offset_3d for x in plate_x], [y + offset_3d for y in plate_y],
            color='#AAAAAA', edgecolor='#888888', linewidth=1, zorder=0)

    # Fixed pivots
    for px in [0, L1]:
        circle = Circle((px, 0), 8, color='#666666', zorder=5)
        ax.add_patch(circle)
        circle_hole = Circle((px, 0), 3, color='#333333', zorder=6)
        ax.add_patch(circle_hole)

    # Position at theta = 60°
    theta = math.radians(60)
    Bx = L2 * math.cos(theta)
    By = L2 * math.sin(theta)

    BD = math.sqrt((Bx - L1)**2 + By**2)
    cos_angle = (BD**2 + L3**2 - L4**2) / (2 * BD * L3)
    cos_angle = max(-1, min(1, cos_angle))
    angle_DBC = math.acos(cos_angle)
    angle_BD = math.atan2(-By, L1 - Bx)
    angle_BC = angle_BD + angle_DBC
    Cx = Bx + L3 * math.cos(angle_BC)
    Cy = By + L3 * math.sin(angle_BC)

    # Draw links as rounded rectangles
    link_width = 10

    # Crank (red)
    ax.plot([0, Bx], [0, By], color='#E74C3C', linewidth=12, solid_capstyle='round', zorder=7)
    ax.plot([0 + 2, Bx + 2], [0 + 2, By + 2], color='#C0392B', linewidth=12,
            solid_capstyle='round', zorder=6)

    # Follower (blue)
    ax.plot([L1, Cx], [0, Cy], color='#3498DB', linewidth=12, solid_capstyle='round', zorder=7)
    ax.plot([L1 + 2, Cx + 2], [0 + 2, Cy + 2], color='#2980B9', linewidth=12,
            solid_capstyle='round', zorder=6)

    # Coupler (green)
    ax.plot([Bx, Cx], [By, Cy], color='#27AE60', linewidth=12, solid_capstyle='round', zorder=8)
    ax.plot([Bx + 2, Cx + 2], [By + 2, Cy + 2], color='#1E8449', linewidth=12,
            solid_capstyle='round', zorder=7)

    # Tracer point
    Px, Py_offset = evolved.Px * scale, evolved.Py * scale
    coupler_len = math.sqrt((Cx - Bx)**2 + (Cy - By)**2)
    ux = (Cx - Bx) / coupler_len
    uy = (Cy - By) / coupler_len
    vx, vy = -uy, ux
    Tx = Bx + (Px / scale) * scale * ux / (L3 / evolved.L3) + Py_offset * vx / (L3 / evolved.L3)
    Ty = By + (Px / scale) * scale * uy / (L3 / evolved.L3) + Py_offset * vy / (L3 / evolved.L3)

    tracer = Circle((Tx, Ty), 5, color='#9B59B6', zorder=10)
    ax.add_patch(tracer)

    # Moving joints
    for jx, jy in [(Bx, By), (Cx, Cy)]:
        joint = Circle((jx, jy), 6, color='white', edgecolor='black', linewidth=2, zorder=9)
        ax.add_patch(joint)

    # Labels with dimensions
    ax.annotate(f'Ground: {L1:.0f}mm', (L1/2, -45), ha='center', fontsize=11)
    ax.annotate(f'Crank: {L2:.0f}mm', (Bx/2 - 15, By/2 + 10), fontsize=10, color='#E74C3C')
    ax.annotate(f'Coupler: {L3:.0f}mm', ((Bx+Cx)/2 + 5, (By+Cy)/2 + 15), fontsize=10, color='#27AE60')
    ax.annotate(f'Follower: {L4:.0f}mm', ((L1+Cx)/2 + 15, Cy/2), fontsize=10, color='#3498DB')
    ax.annotate('Tracer\nPoint', (Tx + 10, Ty), fontsize=10, color='#9B59B6', fontweight='bold')

    ax.set_xlim(-40, L1 + 40)
    ax.set_ylim(-60, 80)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('3D-Printable Evolved Linkage Assembly', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/3d_assembly.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: images/3d_assembly.png")


def plot_parameter_evolution():
    """Show how parameters changed from baseline to evolved."""
    fig, ax = plt.subplots(figsize=(10, 6))

    params = ['L1\n(Ground)', 'L2\n(Crank)', 'L3\n(Coupler)', 'L4\n(Follower)', 'Px\n(Tracer X)', 'Py\n(Tracer Y)']
    baseline = [4.0, 1.5, 3.5, 3.0, 1.75, 1.0]
    evolved = [4.48, 1.23, 4.31, 2.78, 2.618, 1.5]

    x = np.arange(len(params))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, evolved, width, label='Evolved', color='#27AE60', alpha=0.8)

    # Add change percentages
    for i, (b, e) in enumerate(zip(baseline, evolved)):
        change = (e - b) / b * 100
        color = 'green' if change > 0 else 'red'
        ax.annotate(f'{change:+.0f}%', (i + width/2, e + 0.15),
                   ha='center', fontsize=9, color=color, fontweight='bold')

    # Highlight golden ratio
    ax.annotate('φ ≈ 1.618\n(Golden Ratio!)', (4 + width/2, 2.618 + 0.3),
               ha='center', fontsize=10, color='#9B59B6', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#F5EEF8', edgecolor='#9B59B6'))

    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Parameter Evolution: Baseline → Evolved', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/parameter_evolution.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: images/parameter_evolution.png")


def plot_linkage_animation_frames():
    """Create a multi-frame visualization showing linkage motion."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    evolved = load_linkage('evolved_figure8.py')
    target = get_target('figure8', num_points=100)

    # Normalize target
    tx = [p[0] for p in target]
    ty = [p[1] for p in target]
    tcx, tcy = sum(tx)/len(tx), sum(ty)/len(ty)
    tscale = max(max(abs(x-tcx) for x in tx), max(abs(y-tcy) for y in ty))
    tx = [(x-tcx)/tscale for x in tx]
    ty = [(y-tcy)/tscale for y in ty]

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    all_traced = evolved.trace_path(360)

    # Normalize traced path once
    px = [p[0] for p in all_traced]
    py = [p[1] for p in all_traced]
    pcx, pcy = sum(px)/len(px), sum(py)/len(py)
    pscale = max(max(abs(x-pcx) for x in px), max(abs(y-pcy) for y in py))

    for idx, (ax, angle) in enumerate(zip(axes, angles)):
        theta = math.radians(angle)

        # Draw target
        ax.plot(tx, ty, 'r--', linewidth=1, alpha=0.5)

        # Draw traced path up to this point
        frame_idx = int(angle / 360 * len(all_traced))
        traced_so_far = all_traced[:frame_idx+1]
        if traced_so_far:
            tfx = [(p[0]-pcx)/pscale for p in traced_so_far]
            tfy = [(p[1]-pcy)/pscale for p in traced_so_far]
            ax.plot(tfx, tfy, 'b-', linewidth=2)
            ax.plot(tfx[-1], tfy[-1], 'o', color='#9B59B6', markersize=10)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_title(f'θ = {angle}°', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Linkage Motion: Tracing the Figure-8', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/motion_sequence.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: images/motion_sequence.png")


if __name__ == "__main__":
    print("Generating README images...")
    plot_linkage_diagram()
    plot_evolution_comparison()
    plot_fitness_progression()
    plot_3d_assembly()
    plot_parameter_evolution()
    plot_linkage_animation_frames()
    print("\nAll images generated!")
