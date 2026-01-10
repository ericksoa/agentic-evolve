#!/usr/bin/env python3
"""
Linkage Visualizer

Creates visualizations of linkage motion and traced paths.
Outputs PNG images and optionally animated GIFs.
"""

import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

from linkage import Linkage4Bar, load_linkage
from targets import get_target


def plot_linkage_frame(ax, linkage: Linkage4Bar, theta: float, show_tracer_history: list = None):
    """Plot a single frame of the linkage."""
    ax.clear()

    # Fixed pivots
    Ax, Ay = 0, 0
    Dx, Dy = linkage.L1, 0

    # Solve position
    pos = linkage.solve_position(theta)
    if pos is None:
        ax.set_title(f"Invalid position at θ={math.degrees(theta):.1f}°")
        return

    Bx, By, Cx, Cy = pos

    # Get tracer point
    tracer = linkage.get_tracer_point(theta)

    # Draw ground (fixed frame)
    ax.plot([Ax - 0.3, Dx + 0.3], [0, 0], 'k-', linewidth=3, label='Ground')

    # Draw fixed pivots
    ax.plot(Ax, Ay, 'ko', markersize=12, markerfacecolor='gray')
    ax.plot(Dx, Dy, 'ko', markersize=12, markerfacecolor='gray')

    # Draw links
    ax.plot([Ax, Bx], [Ay, By], 'b-', linewidth=4, label='Crank')
    ax.plot([Bx, Cx], [By, Cy], 'g-', linewidth=4, label='Coupler')
    ax.plot([Dx, Cx], [Dy, Cy], 'r-', linewidth=4, label='Follower')

    # Draw joints
    ax.plot(Bx, By, 'ko', markersize=8)
    ax.plot(Cx, Cy, 'ko', markersize=8)

    # Draw tracer point and history
    if tracer:
        Tx, Ty = tracer
        ax.plot(Tx, Ty, 'mo', markersize=10, markerfacecolor='magenta', label='Tracer')

        if show_tracer_history:
            xs = [p[0] for p in show_tracer_history]
            ys = [p[1] for p in show_tracer_history]
            ax.plot(xs, ys, 'm-', linewidth=1, alpha=0.5)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"4-Bar Linkage (θ={math.degrees(theta):.1f}°)")


def plot_comparison(linkage: Linkage4Bar, target_name: str, output_path: str = None):
    """Plot the traced path vs target shape."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get paths
    traced = linkage.trace_path(num_samples=360)
    target = get_target(target_name, num_points=200)

    # Left plot: Traced path
    ax1 = axes[0]
    if traced:
        xs = [p[0] for p in traced]
        ys = [p[1] for p in traced]
        ax1.plot(xs, ys, 'b-', linewidth=2, label='Traced')
        ax1.plot(xs[0], ys[0], 'go', markersize=8, label='Start')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Linkage Traced Path')

    # Right plot: Comparison
    ax2 = axes[1]
    if traced:
        xs = [p[0] for p in traced]
        ys = [p[1] for p in traced]
        ax2.plot(xs, ys, 'b-', linewidth=2, label='Traced', alpha=0.7)

    if target:
        xs = [p[0] for p in target]
        ys = [p[1] for p in target]
        ax2.plot(xs, ys, 'r--', linewidth=2, label=f'Target ({target_name})', alpha=0.7)

    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Traced vs Target')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()


def create_animation(linkage: Linkage4Bar, output_path: str = None, fps: int = 30, duration: float = 3.0):
    """Create an animated visualization of the linkage motion."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required for animation")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    num_frames = int(fps * duration)
    tracer_history = []

    def animate(frame):
        theta = 2 * math.pi * frame / num_frames
        tracer = linkage.get_tracer_point(theta)
        if tracer:
            tracer_history.append(tracer)
        plot_linkage_frame(ax, linkage, theta, tracer_history)

        # Set consistent axis limits
        all_points = tracer_history + [(0, 0), (linkage.L1, 0)]
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            margin = 1.0
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)

    anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps)

    if output_path:
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            anim.save(output_path, fps=fps)
        print(f"Saved animation to {output_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <linkage.py> [--target=<name>] [--animate] [--output=<path>]")
        print("Options:")
        print("  --target=<name>  Target shape to compare (heart, figure8, line, circle, star)")
        print("  --animate        Create animation instead of static plot")
        print("  --output=<path>  Save to file instead of displaying")
        sys.exit(1)

    linkage_path = sys.argv[1]
    animate = "--animate" in sys.argv
    output_path = None
    target_name = "figure8"

    for arg in sys.argv:
        if arg.startswith("--output="):
            output_path = arg.split("=")[1]
        if arg.startswith("--target="):
            target_name = arg.split("=")[1]

    try:
        linkage = load_linkage(linkage_path)
    except Exception as e:
        print(f"Error loading linkage: {e}")
        sys.exit(1)

    if animate:
        create_animation(linkage, output_path)
    else:
        plot_comparison(linkage, target_name, output_path)


if __name__ == "__main__":
    main()
