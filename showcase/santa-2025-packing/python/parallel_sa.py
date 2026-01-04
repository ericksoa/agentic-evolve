#!/usr/bin/env python3
"""
Parallel Simulated Annealing for Santa 2025 Packing - Gen106

Runs multiple SA chains in parallel on GPU.
Each chain operates on a different configuration.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
import time

from gpu_primitives import (
    TreeTensor,
    gpu_transform_trees,
    gpu_compute_bbox,
    gpu_check_bbox_overlaps,
    gpu_score_configs,
    gpu_count_overlaps,
    evaluate_configs,
    get_device,
)


class ParallelSA:
    """
    GPU-accelerated parallel simulated annealing.

    Runs multiple SA chains simultaneously, each on a different configuration.
    Uses batched GPU operations for evaluation.
    """

    def __init__(
        self,
        n_trees: int,
        n_chains: int = 32,
        device: Optional[torch.device] = None,
    ):
        self.n_trees = n_trees
        self.n_chains = n_chains
        self.device = device or get_device()

        self.tree_tensor = TreeTensor(self.device)

        # Current configurations: (n_chains, n_trees, 3)
        self.configs: Optional[torch.Tensor] = None

        # Current fitness: (n_chains,)
        self.fitness: Optional[torch.Tensor] = None

        # Best solution tracking (global across all chains)
        self.best_config: Optional[torch.Tensor] = None
        self.best_fitness: float = float('inf')
        self.best_side_length: float = float('inf')

    def compress_towards_center(self, strength: float = 0.02):
        """Move all trees towards center of bounding box."""
        # Get current bbox center
        _, bbox, _, _ = evaluate_configs(self.tree_tensor, self.configs)
        min_coords = bbox[..., :2].min(dim=1).values  # (n_chains, 2)
        max_coords = bbox[..., 2:].max(dim=1).values  # (n_chains, 2)
        centers = (min_coords + max_coords) / 2  # (n_chains, 2)

        # Move trees towards center
        centers_expanded = centers.unsqueeze(1)  # (n_chains, 1, 2)
        delta = centers_expanded - self.configs[..., :2]  # (n_chains, n_trees, 2)
        self.configs[..., :2] += strength * delta

        # Update fitness
        self.fitness = self._evaluate(self.configs)

    def initialize(self, configs: torch.Tensor):
        """
        Initialize SA chains with configurations.

        Args:
            configs: (n_chains, n_trees, 3) initial configurations
        """
        assert configs.shape[0] == self.n_chains
        assert configs.shape[1] == self.n_trees
        assert configs.shape[2] == 3

        self.configs = configs.clone().to(self.device)
        self.fitness = self._evaluate(self.configs)

    def _evaluate(
        self,
        configs: torch.Tensor,
        overlap_penalty: float = 1000.0
    ) -> torch.Tensor:
        """Evaluate fitness for all configurations."""
        _, bbox, overlaps, side_lengths = evaluate_configs(self.tree_tensor, configs)
        overlap_counts = gpu_count_overlaps(overlaps)
        return side_lengths + overlap_penalty * overlap_counts.float()

    def _get_side_lengths(self, configs: torch.Tensor) -> torch.Tensor:
        """Get just the side lengths (no penalty)."""
        _, bbox, _, side_lengths = evaluate_configs(self.tree_tensor, configs)
        return side_lengths

    def _get_overlaps(self, configs: torch.Tensor) -> torch.Tensor:
        """Get overlap counts."""
        _, bbox, overlaps, _ = evaluate_configs(self.tree_tensor, configs)
        return gpu_count_overlaps(overlaps)

    def step(
        self,
        temperature: float,
        move_scale: float = 0.05,
        angle_scale: float = 5.0,
        boundary_focus: float = 0.5,
        overlap_penalty: float = 1000.0,
    ) -> Tuple[int, int]:
        """
        Perform one SA step across all chains.

        Args:
            temperature: Current temperature
            move_scale: Scale for position moves
            angle_scale: Scale for angle moves (degrees)
            boundary_focus: Probability of moving boundary trees
            overlap_penalty: Penalty per overlapping pair

        Returns:
            (n_accepted, n_improved): Number of accepted and improved moves
        """
        # Select which tree to move in each chain
        # With boundary_focus probability, pick a boundary tree
        # Otherwise, pick any tree

        # For now, random tree selection
        tree_indices = torch.randint(0, self.n_trees, (self.n_chains,), device=self.device)

        # Generate random moves
        # Position: Gaussian with scale proportional to temperature
        pos_delta = torch.randn(self.n_chains, 2, device=self.device) * move_scale
        angle_delta = torch.randn(self.n_chains, device=self.device) * angle_scale

        # Create proposed configs
        proposed = self.configs.clone()

        # Apply moves to selected trees
        batch_indices = torch.arange(self.n_chains, device=self.device)
        proposed[batch_indices, tree_indices, 0] += pos_delta[:, 0]
        proposed[batch_indices, tree_indices, 1] += pos_delta[:, 1]
        proposed[batch_indices, tree_indices, 2] += angle_delta
        proposed[..., 2] = proposed[..., 2] % 360.0

        # Evaluate proposed configs
        proposed_fitness = self._evaluate(proposed, overlap_penalty)

        # Compute acceptance probability
        delta = proposed_fitness - self.fitness
        accept_prob = torch.exp(-delta / max(temperature, 1e-10))
        accept_prob = torch.clamp(accept_prob, 0, 1)

        # Random acceptance
        accept_mask = torch.rand(self.n_chains, device=self.device) < accept_prob

        # Update accepted chains
        n_accepted = accept_mask.sum().item()
        n_improved = (delta < 0).sum().item()

        self.configs[accept_mask] = proposed[accept_mask]
        self.fitness[accept_mask] = proposed_fitness[accept_mask]

        # Update global best
        best_idx = self.fitness.argmin()
        if self.fitness[best_idx].item() < self.best_fitness:
            self.best_fitness = self.fitness[best_idx].item()
            self.best_config = self.configs[best_idx].clone()
            self.best_side_length = self._get_side_lengths(
                self.best_config.unsqueeze(0)
            ).item()

        return n_accepted, n_improved

    def run(
        self,
        n_iterations: int = 10000,
        initial_temp: float = 1.0,
        final_temp: float = 0.001,
        cooling: str = 'exponential',  # or 'linear'
        move_scale: float = 0.05,
        angle_scale: float = 5.0,
        overlap_penalty: float = 1000.0,
        verbose: bool = True,
        log_interval: int = 500,
        compression_interval: int = 0,  # 0 = no compression
        compression_strength: float = 0.02,
    ) -> torch.Tensor:
        """
        Run parallel SA optimization.

        Args:
            n_iterations: Number of SA iterations
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling: Cooling schedule ('exponential' or 'linear')
            move_scale: Base scale for position moves
            angle_scale: Base scale for angle moves
            overlap_penalty: Penalty per overlapping pair
            verbose: Print progress
            log_interval: How often to print progress

        Returns:
            best_config: (n_trees, 3) best configuration found
        """
        if cooling == 'exponential':
            cooling_rate = (final_temp / initial_temp) ** (1.0 / n_iterations)
        else:
            temp_delta = (initial_temp - final_temp) / n_iterations

        temp = initial_temp
        start_time = time.perf_counter()

        total_accepted = 0
        total_improved = 0

        for i in range(n_iterations):
            # Adaptive move scale (larger at high temp)
            adaptive_move = move_scale * (temp / initial_temp + 0.2)
            adaptive_angle = angle_scale * (temp / initial_temp + 0.2)

            n_acc, n_imp = self.step(
                temperature=temp,
                move_scale=adaptive_move,
                angle_scale=adaptive_angle,
                overlap_penalty=overlap_penalty,
            )

            total_accepted += n_acc
            total_improved += n_imp

            # Cool down
            if cooling == 'exponential':
                temp *= cooling_rate
            else:
                temp -= temp_delta

            # Compression phase
            if compression_interval > 0 and (i + 1) % compression_interval == 0:
                self.compress_towards_center(compression_strength)

            # Logging
            if verbose and (i + 1) % log_interval == 0:
                elapsed = time.perf_counter() - start_time
                accept_rate = total_accepted / (log_interval * self.n_chains)
                improve_rate = total_improved / (log_interval * self.n_chains)

                # Get current stats
                side_lengths = self._get_side_lengths(self.configs)
                overlaps = self._get_overlaps(self.configs)

                print(f"Iter {i+1:5d}: T={temp:.4f} "
                      f"best={self.best_side_length:.4f} "
                      f"mean_side={side_lengths.mean().item():.4f} "
                      f"mean_overlaps={overlaps.float().mean().item():.1f} "
                      f"accept={accept_rate:.1%} "
                      f"[{elapsed:.1f}s]")

                total_accepted = 0
                total_improved = 0

        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"\nCompleted in {elapsed:.1f}s")
            print(f"Best side length: {self.best_side_length:.4f}")

            # Final overlap check
            best_overlaps = self._get_overlaps(self.best_config.unsqueeze(0)).item()
            print(f"Best overlaps: {best_overlaps}")

        return self.best_config


class HybridOptimizer:
    """
    Combines population-based evolution with parallel SA.

    Strategy:
    1. Use evolutionary operators for diversity and global search
    2. Use SA for local refinement
    3. Share best solutions between methods
    """

    def __init__(
        self,
        n_trees: int,
        pop_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.n_trees = n_trees
        self.pop_size = pop_size
        self.device = device or get_device()

        self.tree_tensor = TreeTensor(self.device)
        self.sa = ParallelSA(n_trees, n_chains=pop_size, device=self.device)

        self.best_config: Optional[torch.Tensor] = None
        self.best_fitness: float = float('inf')
        self.best_side_length: float = float('inf')

    def run(
        self,
        initial_configs: torch.Tensor,
        n_generations: int = 20,
        sa_iterations_per_gen: int = 500,
        initial_temp: float = 0.5,
        final_temp: float = 0.01,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Run hybrid optimization.

        Each generation:
        1. Run SA on current population
        2. Keep elite, reseed rest from best + mutations

        Args:
            initial_configs: (pop_size, n_trees, 3) starting configurations
            n_generations: Number of evolution generations
            sa_iterations_per_gen: SA iterations per generation
            initial_temp: SA initial temperature
            final_temp: SA final temperature
            verbose: Print progress

        Returns:
            best_config: (n_trees, 3) best found
        """
        self.sa.initialize(initial_configs)

        for gen in range(n_generations):
            # Scale temperature per generation
            gen_initial_temp = initial_temp * (0.9 ** gen)
            gen_final_temp = max(final_temp, gen_initial_temp * 0.1)

            if verbose:
                print(f"\n=== Generation {gen+1}/{n_generations} ===")
                print(f"Temperature: {gen_initial_temp:.4f} -> {gen_final_temp:.4f}")

            # Run SA
            best = self.sa.run(
                n_iterations=sa_iterations_per_gen,
                initial_temp=gen_initial_temp,
                final_temp=gen_final_temp,
                verbose=verbose,
                log_interval=sa_iterations_per_gen // 2,
            )

            # Update global best
            if self.sa.best_fitness < self.best_fitness:
                self.best_fitness = self.sa.best_fitness
                self.best_side_length = self.sa.best_side_length
                self.best_config = self.sa.best_config.clone()

            # Reseed: keep top 25%, fill rest with mutations of best
            fitness = self.sa.fitness
            sorted_indices = torch.argsort(fitness)
            n_elite = self.pop_size // 4

            elite_configs = self.sa.configs[sorted_indices[:n_elite]]

            # Create new population
            new_configs = torch.zeros_like(self.sa.configs)
            new_configs[:n_elite] = elite_configs

            # Fill rest with mutated versions of global best
            for i in range(n_elite, self.pop_size):
                base = self.best_config.clone()
                # Add noise
                noise_pos = torch.randn(self.n_trees, 2, device=self.device) * 0.05
                noise_angle = torch.randn(self.n_trees, device=self.device) * 5.0
                base[:, :2] += noise_pos
                base[:, 2] += noise_angle
                base[:, 2] = base[:, 2] % 360.0
                new_configs[i] = base

            self.sa.initialize(new_configs)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Final Results")
            print(f"{'='*60}")
            print(f"Best side length: {self.best_side_length:.4f}")

            # Check overlaps
            _, _, overlaps, _ = evaluate_configs(
                self.tree_tensor,
                self.best_config.unsqueeze(0)
            )
            n_overlaps = gpu_count_overlaps(overlaps).item()
            print(f"Best overlaps: {n_overlaps}")

        return self.best_config


def generate_grid_config(n: int, box_size: float) -> np.ndarray:
    """Generate a grid-based initial configuration."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    spacing = box_size / max(rows, cols)

    config = np.zeros((n, 3), dtype=np.float32)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = (col - (cols - 1) / 2) * spacing
            y = (row - (rows - 1) / 2) * spacing
            angle = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            config[idx] = [x, y, angle]
            idx += 1

    return config


def test_parallel_sa(n_trees: int = 20, n_chains: int = 16, n_iterations: int = 2000):
    """Test parallel SA."""
    print(f"Parallel SA Test (n={n_trees}, chains={n_chains}, iters={n_iterations})")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Generate initial configs
    box_size = np.sqrt(n_trees * 0.35 / 0.6) * 1.2
    initial_configs = [generate_grid_config(n_trees, box_size) for _ in range(n_chains)]
    initial_tensor = torch.tensor(np.stack(initial_configs), dtype=torch.float32, device=device)

    sa = ParallelSA(n_trees=n_trees, n_chains=n_chains, device=device)
    sa.initialize(initial_tensor)

    best = sa.run(
        n_iterations=n_iterations,
        initial_temp=0.5,
        final_temp=0.001,
        move_scale=0.08,
        angle_scale=10.0,
        overlap_penalty=1000.0,
        verbose=True,
        log_interval=500,
    )

    return sa


def test_hybrid(n_trees: int = 20, pop_size: int = 32):
    """Test hybrid optimizer."""
    print(f"\nHybrid Optimizer Test (n={n_trees}, pop={pop_size})")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Generate initial configs
    box_size = np.sqrt(n_trees * 0.35 / 0.6) * 1.2
    initial_configs = [generate_grid_config(n_trees, box_size) for _ in range(pop_size)]
    initial_tensor = torch.tensor(np.stack(initial_configs), dtype=torch.float32, device=device)

    hybrid = HybridOptimizer(n_trees=n_trees, pop_size=pop_size, device=device)
    best = hybrid.run(
        initial_configs=initial_tensor,
        n_generations=10,
        sa_iterations_per_gen=500,
        verbose=True,
    )

    return hybrid


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'hybrid':
        test_hybrid(n_trees=20, pop_size=32)
    else:
        test_parallel_sa(n_trees=20, n_chains=32, n_iterations=3000)
