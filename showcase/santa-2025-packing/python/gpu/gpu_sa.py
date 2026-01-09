#!/usr/bin/env python3
"""
GPU-Accelerated Simulated Annealing for Tree Packing

Runs thousands of parallel SA chains on GPU to explore the solution space
much more thoroughly than CPU-based approaches.

Key optimizations:
1. Batch processing: Run B independent SA chains in parallel
2. Vectorized operations: Position/angle updates, bounding box computation
3. Efficient overlap checking: Bounding box filter + segment intersection

Designed for NVIDIA L40S (48GB VRAM) but works on any CUDA GPU.
"""

import torch
import torch.nn.functional as F
import math
import time
import argparse
import csv
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import numpy as np

# Tree polygon vertices (15 vertices)
TREE_VERTICES = torch.tensor([
    [0.0, 0.8],      # Tip
    [0.125, 0.5],    # Right top tier outer
    [0.0625, 0.5],   # Right top tier inner
    [0.2, 0.25],     # Right mid tier outer
    [0.1, 0.25],     # Right mid tier inner
    [0.35, 0.0],     # Right base outer
    [0.075, 0.0],    # Right trunk
    [0.075, -0.2],   # Right trunk bottom
    [-0.075, -0.2],  # Left trunk bottom
    [-0.075, 0.0],   # Left trunk
    [-0.35, 0.0],    # Left base outer
    [-0.1, 0.25],    # Left mid tier inner
    [-0.2, 0.25],    # Left mid tier outer
    [-0.0625, 0.5],  # Left top tier inner
    [-0.125, 0.5],   # Left top tier outer
], dtype=torch.float32)

NUM_VERTICES = 15


def compute_side_from_solution(trees: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side from a solution."""
    all_xs, all_ys = [], []
    for x, y, deg in trees:
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        for vx, vy in TREE_VERTICES.tolist():
            rx = vx * cos_a - vy * sin_a + x
            ry = vx * sin_a + vy * cos_a + y
            all_xs.append(rx)
            all_ys.append(ry)
    return max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys))


def load_best_from_csv(csv_path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    Load best solutions from submission CSV.

    Returns:
        Dict mapping n -> list of (x, y, angle) tuples for each tree
    """
    solutions = {}

    if not Path(csv_path).exists():
        return solutions

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parts = row['id'].split('_')
            n = int(parts[0])

            x = float(row['x'][1:])  # Remove 's' prefix
            y = float(row['y'][1:])
            deg = float(row['deg'][1:])

            if n not in solutions:
                solutions[n] = []
            solutions[n].append((x, y, deg))

    return solutions


def get_device(force_cpu: bool = False):
    """Get the best available device."""
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS can be unstable for some operations, use with caution
    # elif torch.backends.mps.is_available():
    #     return torch.device('mps')
    return torch.device('cpu')


def rotate_vertices(vertices: torch.Tensor, angles_rad: torch.Tensor) -> torch.Tensor:
    """
    Rotate vertices by angles.

    Args:
        vertices: [V, 2] base vertices
        angles_rad: [B, N] angles in radians for B batches, N trees

    Returns:
        [B, N, V, 2] rotated vertices
    """
    B, N = angles_rad.shape
    V = vertices.shape[0]

    cos_a = torch.cos(angles_rad)  # [B, N]
    sin_a = torch.sin(angles_rad)  # [B, N]

    # Rotation matrices [B, N, 2, 2]
    rot = torch.stack([
        torch.stack([cos_a, -sin_a], dim=-1),
        torch.stack([sin_a, cos_a], dim=-1)
    ], dim=-2)

    # Expand vertices for batch matmul: [B, N, V, 2]
    v = vertices.unsqueeze(0).unsqueeze(0).expand(B, N, V, 2)

    # Apply rotation: [B, N, V, 2]
    # rot[b,n] @ v[b,n,v] for each vertex v
    rotated = torch.einsum('bnij,bnvj->bnvi', rot, v)

    return rotated


def translate_vertices(rotated: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Translate rotated vertices by positions.

    Args:
        rotated: [B, N, V, 2] rotated vertices
        positions: [B, N, 2] tree positions (x, y)

    Returns:
        [B, N, V, 2] translated vertices
    """
    return rotated + positions.unsqueeze(2)


def compute_bounding_boxes(vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bounding boxes for all trees.

    Args:
        vertices: [B, N, V, 2] tree vertices

    Returns:
        mins: [B, N, 2] min x, y for each tree
        maxs: [B, N, 2] max x, y for each tree
    """
    mins = vertices.min(dim=2)[0]  # [B, N, 2]
    maxs = vertices.max(dim=2)[0]  # [B, N, 2]
    return mins, maxs


def compute_side_lengths(mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    """
    Compute bounding square side length for each batch.

    Args:
        mins: [B, N, 2] min corners
        maxs: [B, N, 2] max corners

    Returns:
        [B] side lengths
    """
    # Global min/max across all trees
    global_mins = mins.min(dim=1)[0]  # [B, 2]
    global_maxs = maxs.max(dim=1)[0]  # [B, 2]

    widths = global_maxs[:, 0] - global_mins[:, 0]  # [B]
    heights = global_maxs[:, 1] - global_mins[:, 1]  # [B]

    return torch.maximum(widths, heights)


def boxes_overlap(mins1: torch.Tensor, maxs1: torch.Tensor,
                  mins2: torch.Tensor, maxs2: torch.Tensor,
                  margin: float = 1e-6) -> torch.Tensor:
    """
    Check if bounding boxes overlap.

    Args:
        mins1, maxs1: [B, 2] first box corners
        mins2, maxs2: [B, 2] second box corners
        margin: safety margin

    Returns:
        [B] boolean tensor
    """
    # No overlap if separated in x or y
    sep_x = (maxs1[:, 0] + margin < mins2[:, 0]) | (maxs2[:, 0] + margin < mins1[:, 0])
    sep_y = (maxs1[:, 1] + margin < mins2[:, 1]) | (maxs2[:, 1] + margin < mins1[:, 1])

    return ~(sep_x | sep_y)


def ccw(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Counter-clockwise test for points."""
    return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) - \
           (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])


def segments_intersect(A: torch.Tensor, B: torch.Tensor,
                       C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Check if segment AB intersects segment CD (proper intersection).

    Args:
        A, B, C, D: [..., 2] point tensors

    Returns:
        [...] boolean tensor
    """
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)

    return ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))


def point_in_polygon(point: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    Ray casting algorithm to check if point is inside polygon.

    Args:
        point: [..., 2] point coordinates
        polygon: [V, 2] polygon vertices

    Returns:
        [...] boolean tensor
    """
    x, y = point[..., 0], point[..., 1]
    n = polygon.shape[0]
    inside = torch.zeros_like(x, dtype=torch.bool)

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i, 0], polygon[i, 1]
        xj, yj = polygon[j, 0], polygon[j, 1]

        cond1 = (yi > y) != (yj > y)
        cond2 = x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi

        inside = inside ^ (cond1 & cond2)
        j = i

    return inside


def check_pair_overlap_detailed(verts1: torch.Tensor, verts2: torch.Tensor) -> bool:
    """
    Detailed overlap check between two polygons (CPU, single pair).

    Args:
        verts1: [V, 2] first polygon vertices
        verts2: [V, 2] second polygon vertices

    Returns:
        True if polygons overlap
    """
    V = verts1.shape[0]

    # Check edge intersections
    for i in range(V):
        j = (i + 1) % V
        a1, a2 = verts1[i], verts1[j]

        for k in range(V):
            l = (k + 1) % V
            b1, b2 = verts2[k], verts2[l]

            if segments_intersect(a1, a2, b1, b2).item():
                return True

    # Check if any vertex of poly1 is inside poly2
    for i in range(V):
        if point_in_polygon(verts1[i], verts2).item():
            return True

    # Check if any vertex of poly2 is inside poly1
    for i in range(V):
        if point_in_polygon(verts2[i], verts1).item():
            return True

    return False


def check_overlaps_batch(vertices: torch.Tensor,
                         chain_idx: int,
                         tree_idx: int,
                         fast_mode: bool = True,
                         margin: float = None) -> bool:
    """
    Check if tree at tree_idx overlaps with any other tree in chain.

    Args:
        vertices: [B, N, V, 2] all vertices
        chain_idx: which chain to check
        tree_idx: which tree in that chain
        fast_mode: If True, use only bounding box check (faster)
        margin: Safety margin for bounding box check. Default: 0.001 for fast, 1e-6 for exact.

    Returns:
        True if overlap exists
    """
    N = vertices.shape[1]
    verts_target = vertices[chain_idx, tree_idx]  # [V, 2]

    # Get bounding box of target
    min_target = verts_target.min(dim=0)[0]
    max_target = verts_target.max(dim=0)[0]

    # Tight margin to be conservative (reject moves that might cause overlap)
    # 0.001 is small enough to not reject many valid moves but catches edge cases
    if margin is None:
        margin = 0.001 if fast_mode else 1e-6

    for i in range(N):
        if i == tree_idx:
            continue

        verts_other = vertices[chain_idx, i]  # [V, 2]

        # Quick bounding box check
        min_other = verts_other.min(dim=0)[0]
        max_other = verts_other.max(dim=0)[0]

        # Check for bounding box overlap with margin
        if (max_target[0] + margin < min_other[0] or max_other[0] + margin < min_target[0] or
            max_target[1] + margin < min_other[1] or max_other[1] + margin < min_target[1]):
            continue

        # In fast mode, bounding box overlap = overlap (conservative)
        if fast_mode:
            return True

        # Detailed check only in non-fast mode
        if check_pair_overlap_detailed(verts_target, verts_other):
            return True

    return False


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing."""
    iterations: int = 10000
    initial_temp: float = 1.0
    cooling_rate: float = 0.9995
    translation_small: float = 0.05  # Small move for fine-tuning
    translation_large: float = 0.15  # Larger move for exploration
    rotation_max: float = 30.0  # degrees - reduced for refinement
    strict_overlap: bool = False  # If True, use exact polygon overlap check (slow but exact)
    overlap_margin: float = 0.001  # Bounding box safety margin (smaller = more conservative)


class GPUSimulatedAnnealing:
    """GPU-accelerated parallel Simulated Annealing."""

    def __init__(self, n_trees: int, n_chains: int, config: SAConfig = None, device=None,
                 best_solutions: Dict[int, List[Tuple[float, float, float]]] = None):
        """
        Initialize GPU SA.

        Args:
            n_trees: Number of trees to pack
            n_chains: Number of parallel SA chains
            config: SA configuration
            device: torch device
            best_solutions: Optional dict mapping n -> list of (x, y, angle) tuples
        """
        self.n = n_trees
        self.B = n_chains
        self.config = config or SAConfig()
        self.device = device or get_device()
        self.best_solutions = best_solutions or {}

        # Move base vertices to device
        self.base_vertices = TREE_VERTICES.to(self.device)

        # State: positions [B, N, 2] and angles [B, N]
        self.positions = None
        self.angles = None

    def initialize_greedy(self):
        """Initialize with randomized compact placement for global search."""
        B, N = self.B, self.n
        device = self.device

        # Initialize arrays
        self.positions = torch.zeros(B, N, 2, device=device)
        self.angles = torch.zeros(B, N, device=device)

        if N == 0:
            return

        # Each chain gets different random configuration
        for b in range(B):
            # Random angles with bias towards cardinal directions
            if torch.rand(1).item() < 0.7:
                # 70% chance: cardinal angles (0, 90, 180, 270)
                angle_choices = torch.tensor([0.0, 90.0, 180.0, 270.0], device=device)
                self.angles[b] = angle_choices[torch.randint(4, (N,), device=device)]
            else:
                # 30% chance: random angles
                self.angles[b] = torch.rand(N, device=device) * 360

            # Randomized placement strategy per chain
            strategy = torch.randint(3, (1,)).item()

            if strategy == 0:
                # Grid with random offset
                grid_size = int(math.ceil(math.sqrt(N)))
                spacing = 0.8 + torch.rand(1).item() * 0.3  # 0.8-1.1
                offset_x = (torch.rand(1).item() - 0.5) * 0.3
                offset_y = (torch.rand(1).item() - 0.5) * 0.3
                for i in range(N):
                    row = i // grid_size
                    col = i % grid_size
                    self.positions[b, i, 0] = col * spacing + offset_x
                    self.positions[b, i, 1] = row * spacing + offset_y
            elif strategy == 1:
                # Hexagonal packing
                spacing = 0.85 + torch.rand(1).item() * 0.2
                for i in range(N):
                    row = i // int(math.ceil(math.sqrt(N)))
                    col = i % int(math.ceil(math.sqrt(N)))
                    x_offset = 0.5 * spacing if row % 2 == 1 else 0
                    self.positions[b, i, 0] = col * spacing + x_offset
                    self.positions[b, i, 1] = row * spacing * 0.866  # sqrt(3)/2
            else:
                # Random compact placement
                scale = math.sqrt(N) * 0.5
                self.positions[b, :, 0] = (torch.rand(N, device=device) - 0.5) * scale
                self.positions[b, :, 1] = (torch.rand(N, device=device) - 0.5) * scale

            # Add small noise for variation
            noise = (torch.rand(N, 2, device=device) - 0.5) * 0.15
            self.positions[b] += noise

    def initialize_from_best(self, noise_levels: List[float] = None):
        """
        Initialize from current best solution with varying noise levels.

        Each chain starts from the best known solution with different amounts
        of noise to explore the neighborhood while maintaining diversity.

        Args:
            noise_levels: List of noise magnitudes for each chain group.
                         Default: Very small noise to stay near valid region
        """
        B, N = self.B, self.n
        device = self.device

        # Very small noise levels - the best solutions are tightly optimized
        # so we need to stay close to avoid creating overlaps
        if noise_levels is None:
            noise_levels = [0.0, 0.0, 0.001, 0.002, 0.005, 0.01]

        # Check if we have a best solution for this n
        if N not in self.best_solutions or len(self.best_solutions[N]) != N:
            print(f"  No best solution for n={N}, falling back to greedy init")
            self.initialize_greedy()
            return

        # Load best solution
        best = self.best_solutions[N]
        base_positions = torch.tensor([[t[0], t[1]] for t in best], dtype=torch.float32, device=device)
        base_angles = torch.tensor([t[2] for t in best], dtype=torch.float32, device=device)

        # Initialize arrays
        self.positions = torch.zeros(B, N, 2, device=device)
        self.angles = torch.zeros(B, N, device=device)

        # Assign chains to noise level groups
        chains_per_level = max(1, B // len(noise_levels))

        for b in range(B):
            level_idx = min(b // chains_per_level, len(noise_levels) - 1)
            noise_mag = noise_levels[level_idx]

            # Copy base solution
            self.positions[b] = base_positions.clone()
            self.angles[b] = base_angles.clone()

            # Add position noise
            if noise_mag > 0:
                pos_noise = (torch.rand(N, 2, device=device) - 0.5) * 2 * noise_mag
                self.positions[b] += pos_noise

                # Add angle noise (proportional to position noise)
                angle_noise = (torch.rand(N, device=device) - 0.5) * 2 * noise_mag * 180
                self.angles[b] = (self.angles[b] + angle_noise) % 360

    def get_vertices(self) -> torch.Tensor:
        """Get all tree vertices. Returns [B, N, V, 2]."""
        angles_rad = self.angles * math.pi / 180.0
        rotated = rotate_vertices(self.base_vertices, angles_rad)
        return translate_vertices(rotated, self.positions)

    def get_side_lengths(self) -> torch.Tensor:
        """Get side lengths for all chains. Returns [B]."""
        verts = self.get_vertices()
        mins, maxs = compute_bounding_boxes(verts)
        return compute_side_lengths(mins, maxs)

    def run_sa_step(self, temp: float) -> Tuple[torch.Tensor, int]:
        """
        Run one SA step for all chains in parallel.

        Returns:
            (new_side_lengths, num_accepted)
        """
        B, N = self.B, self.n
        device = self.device
        cfg = self.config

        # Current state
        current_sides = self.get_side_lengths()

        # Pick random tree for each chain
        tree_indices = torch.randint(N, (B,), device=device)

        # Pick perturbation type: 0=small trans, 1=rotation, 2=large trans
        perturb_type = torch.randint(3, (B,), device=device)

        # Save old state
        old_positions = self.positions.clone()
        old_angles = self.angles.clone()

        # Apply perturbations
        for b in range(B):
            idx = tree_indices[b].item()
            pt = perturb_type[b].item()

            if pt == 0:  # Small translation
                dx = (torch.rand(1, device=device).item() - 0.5) * 2 * cfg.translation_small
                dy = (torch.rand(1, device=device).item() - 0.5) * 2 * cfg.translation_small
                self.positions[b, idx, 0] += dx
                self.positions[b, idx, 1] += dy
            elif pt == 1:  # Rotation
                da = (torch.rand(1, device=device).item() - 0.5) * 2 * cfg.rotation_max
                self.angles[b, idx] = (self.angles[b, idx] + da) % 360.0
            else:  # Large translation
                dx = (torch.rand(1, device=device).item() - 0.5) * 2 * cfg.translation_large
                dy = (torch.rand(1, device=device).item() - 0.5) * 2 * cfg.translation_large
                self.positions[b, idx, 0] += dx
                self.positions[b, idx, 1] += dy

        # Check validity and accept/reject
        new_verts = self.get_vertices()
        new_sides = self.get_side_lengths()

        num_accepted = 0
        for b in range(B):
            idx = tree_indices[b].item()

            # Check for overlaps - use strict mode if configured
            has_overlap = check_overlaps_batch(
                new_verts, b, idx,
                fast_mode=not cfg.strict_overlap,
                margin=cfg.overlap_margin
            )

            if has_overlap:
                # Reject - restore old state
                self.positions[b] = old_positions[b]
                self.angles[b] = old_angles[b]
            else:
                # Check acceptance criterion
                delta = new_sides[b] - current_sides[b]

                if delta < 0 or torch.rand(1, device=device) < torch.exp(-delta / temp):
                    # Accept
                    num_accepted += 1
                else:
                    # Reject
                    self.positions[b] = old_positions[b]
                    self.angles[b] = old_angles[b]

        return self.get_side_lengths(), num_accepted

    def run(self, verbose: bool = True, use_best_init: bool = True,
            validate_final: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run full SA optimization.

        Args:
            verbose: Print progress
            use_best_init: If True and best_solutions available, initialize from best
            validate_final: If True, do exact validation at end and mark invalid chains

        Returns:
            (best_positions, best_angles, best_side_lengths) for each chain
            Note: Invalid chains have side_length set to inf
        """
        cfg = self.config

        # Initialize - prefer from best solution if available
        if use_best_init and self.n in self.best_solutions:
            if verbose:
                print(f"Initializing {self.B} chains from best solution for n={self.n}...")
            self.initialize_from_best()
        else:
            if verbose:
                print(f"Initializing {self.B} chains with greedy placement for n={self.n}...")
            self.initialize_greedy()

        # Track best per chain
        best_positions = self.positions.clone()
        best_angles = self.angles.clone()
        best_sides = self.get_side_lengths()

        if verbose:
            print(f"Initial best: {best_sides.min().item():.6f}")

        # SA loop
        temp = cfg.initial_temp
        start_time = time.time()

        for iteration in range(cfg.iterations):
            new_sides, num_accepted = self.run_sa_step(temp)

            # Update best per chain
            improved = new_sides < best_sides
            if improved.any():
                for b in range(self.B):
                    if improved[b]:
                        best_positions[b] = self.positions[b].clone()
                        best_angles[b] = self.angles[b].clone()
                        best_sides[b] = new_sides[b]

            temp *= cfg.cooling_rate

            # Progress
            if verbose and (iteration + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                iter_per_sec = (iteration + 1) / elapsed
                print(f"  Iter {iteration+1}/{cfg.iterations}: "
                      f"best={best_sides.min().item():.6f}, "
                      f"accepted={num_accepted}/{self.B}, "
                      f"temp={temp:.6f}, "
                      f"{iter_per_sec:.1f} iter/s")

        elapsed = time.time() - start_time
        if verbose:
            print(f"Completed in {elapsed:.1f}s ({cfg.iterations/elapsed:.1f} iter/s)")
            print(f"Best across all chains (pre-validation): {best_sides.min().item():.6f}")

        # Final validation pass - mark invalid chains with inf
        if validate_final:
            valid_count = 0
            for b in range(self.B):
                is_valid, overlaps = validate_packing(
                    best_positions[b], best_angles[b], self.base_vertices
                )
                if not is_valid:
                    best_sides[b] = float('inf')  # Mark invalid
                else:
                    valid_count += 1
            if verbose:
                print(f"Validation: {valid_count}/{self.B} chains valid")
                if valid_count > 0:
                    valid_best = best_sides[best_sides < float('inf')].min().item()
                    print(f"Best valid: {valid_best:.6f}")

        return best_positions, best_angles, best_sides


def validate_packing(positions: torch.Tensor, angles: torch.Tensor,
                     base_vertices: torch.Tensor) -> Tuple[bool, int]:
    """
    Validate a packing has no overlaps.

    Returns:
        (is_valid, num_overlaps)
    """
    N = positions.shape[0]
    angles_rad = angles * math.pi / 180.0

    # Compute all vertices
    rotated = rotate_vertices(base_vertices, angles_rad.unsqueeze(0))
    verts = translate_vertices(rotated, positions.unsqueeze(0))[0]  # [N, V, 2]

    num_overlaps = 0
    for i in range(N):
        for j in range(i + 1, N):
            if check_pair_overlap_detailed(verts[i].cpu(), verts[j].cpu()):
                num_overlaps += 1

    return num_overlaps == 0, num_overlaps


def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated SA for tree packing')
    parser.add_argument('n', type=int, help='Number of trees')
    parser.add_argument('--chains', type=int, default=100, help='Number of parallel SA chains')
    parser.add_argument('--iterations', type=int, default=10000, help='SA iterations')
    parser.add_argument('--temp', type=float, default=1.0, help='Initial temperature')
    parser.add_argument('--cooling', type=float, default=0.9995, help='Cooling rate')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--best-csv', type=str, default='submission_best.csv',
                        help='CSV file with best solutions to initialize from')
    parser.add_argument('--greedy-init', action='store_true',
                        help='Use greedy init instead of loading from best')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load best solutions
    best_solutions = {}
    if not args.greedy_init:
        csv_path = Path(__file__).parent.parent.parent / args.best_csv
        if csv_path.exists():
            print(f"Loading best solutions from {args.best_csv}...")
            best_solutions = load_best_from_csv(str(csv_path))
            print(f"  Loaded solutions for {len(best_solutions)} n values")
        else:
            print(f"Warning: {args.best_csv} not found, using greedy init")

    config = SAConfig(
        iterations=args.iterations,
        initial_temp=args.temp,
        cooling_rate=args.cooling
    )

    print(f"\nRunning GPU SA for n={args.n} with {args.chains} chains")
    print(f"Config: {args.iterations} iterations, temp={args.temp}, cooling={args.cooling}")
    if args.n in best_solutions:
        print(f"Init: from best solution (side={compute_side_from_solution(best_solutions[args.n])})")
    else:
        print(f"Init: greedy grid placement")

    sa = GPUSimulatedAnnealing(args.n, args.chains, config, device, best_solutions)
    best_pos, best_ang, best_sides = sa.run(verbose=True, use_best_init=not args.greedy_init)

    # Find overall best
    best_idx = best_sides.argmin().item()
    best_side = best_sides[best_idx].item()

    print(f"\nBest result: chain {best_idx}, side={best_side:.6f}")
    print(f"Score contribution: {best_side**2 / args.n:.6f}")

    # Validate
    is_valid, num_overlaps = validate_packing(
        best_pos[best_idx], best_ang[best_idx], TREE_VERTICES.to(device)
    )
    print(f"Validation: valid={is_valid}, overlaps={num_overlaps}")

    if args.output:
        import json
        result = {
            'n': args.n,
            'side': best_side,
            'score_contribution': best_side**2 / args.n,
            'valid': is_valid,
            'chains': args.chains,
            'iterations': args.iterations,
            'trees': [
                {
                    'x': best_pos[best_idx, i, 0].item(),
                    'y': best_pos[best_idx, i, 1].item(),
                    'angle': best_ang[best_idx, i].item()
                }
                for i in range(args.n)
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
