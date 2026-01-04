#!/usr/bin/env python3
"""
Population-Based Optimization for Santa 2025 Packing - Gen106

Uses GPU-accelerated primitives for batch evaluation.
Implements evolutionary operators: selection, mutation, crossover.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, List
import time

from gpu_primitives import (
    TreeTensor,
    gpu_transform_trees,
    gpu_compute_bbox,
    gpu_check_bbox_overlaps,
    gpu_score_configs,
    gpu_count_overlaps,
    gpu_fitness,
    evaluate_configs,
    get_device,
    TREE_VERTICES_NP,
)


class PopulationOptimizer:
    """
    GPU-accelerated population-based optimizer for tree packing.

    Strategy:
    1. Initialize population from greedy seeds or random
    2. Evaluate all configs in parallel on GPU
    3. Apply evolutionary operators (selection, mutation, crossover)
    4. Repeat until convergence
    """

    def __init__(
        self,
        n_trees: int,
        pop_size: int = 64,
        device: Optional[torch.device] = None,
        overlap_penalty: float = 100.0,
    ):
        self.n_trees = n_trees
        self.pop_size = pop_size
        self.device = device or get_device()
        self.overlap_penalty = overlap_penalty

        self.tree_tensor = TreeTensor(self.device)

        # Population: (pop_size, n_trees, 3) - [x, y, angle_deg]
        self.population: Optional[torch.Tensor] = None

        # Best solution tracking
        self.best_config: Optional[torch.Tensor] = None
        self.best_fitness: float = float('inf')
        self.best_side_length: float = float('inf')

        # Statistics
        self.generation = 0
        self.history: List[dict] = []

    def initialize_random(self, box_size: float = 5.0, seed: Optional[int] = None):
        """Initialize population with random configurations."""
        if seed is not None:
            torch.manual_seed(seed)

        self.population = self.tree_tensor.random_configs(
            self.pop_size, self.n_trees, box_size=box_size
        )
        self.generation = 0

    def initialize_from_greedy(self, greedy_configs: List[np.ndarray]):
        """
        Initialize population from greedy solutions.

        Args:
            greedy_configs: List of numpy arrays, each (n_trees, 3)
        """
        # Convert to tensor
        configs = [torch.tensor(c, dtype=torch.float32) for c in greedy_configs]

        # Pad to population size with variations
        if len(configs) < self.pop_size:
            # Add mutated versions of existing configs
            while len(configs) < self.pop_size:
                base = configs[len(configs) % len(greedy_configs)].clone()
                # Add noise
                noise = torch.randn_like(base) * 0.1
                noise[..., 2] = noise[..., 2] * 10  # Larger angle noise
                configs.append(base + noise)

        # Stack and move to device
        self.population = torch.stack(configs[:self.pop_size]).to(self.device)
        self.generation = 0

    def evaluate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate all configurations in the population.

        Returns:
            fitness: (pop_size,) - fitness scores (lower is better)
            side_lengths: (pop_size,) - side lengths
            overlap_counts: (pop_size,) - number of overlapping pairs
        """
        _, bbox, overlaps, side_lengths = evaluate_configs(
            self.tree_tensor, self.population
        )

        overlap_counts = gpu_count_overlaps(overlaps)
        fitness = side_lengths + self.overlap_penalty * overlap_counts.float()

        return fitness, side_lengths, overlap_counts

    def select_elite(
        self,
        fitness: torch.Tensor,
        elite_ratio: float = 0.25
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top configurations by fitness.

        Args:
            fitness: (pop_size,) fitness scores
            elite_ratio: Fraction of population to keep

        Returns:
            elite_configs: (n_elite, n_trees, 3)
            elite_indices: (n_elite,) indices of elite in original population
        """
        n_elite = max(1, int(self.pop_size * elite_ratio))

        # Sort by fitness (lower is better)
        sorted_indices = torch.argsort(fitness)
        elite_indices = sorted_indices[:n_elite]

        elite_configs = self.population[elite_indices]

        return elite_configs, elite_indices

    def mutate(
        self,
        configs: torch.Tensor,
        position_std: float = 0.05,
        angle_std: float = 5.0,
        mutation_prob: float = 0.3
    ) -> torch.Tensor:
        """
        Apply random mutations to configurations.

        Args:
            configs: (batch, n_trees, 3) configurations
            position_std: Standard deviation for position noise
            angle_std: Standard deviation for angle noise (degrees)
            mutation_prob: Probability of mutating each tree

        Returns:
            mutated: (batch, n_trees, 3) mutated configurations
        """
        mutated = configs.clone()

        # Create mutation mask
        mask = torch.rand(configs.shape[0], configs.shape[1], device=self.device) < mutation_prob

        # Position noise
        pos_noise = torch.randn(configs.shape[0], configs.shape[1], 2, device=self.device) * position_std
        mutated[..., :2] += pos_noise * mask.unsqueeze(-1)

        # Angle noise
        angle_noise = torch.randn(configs.shape[0], configs.shape[1], device=self.device) * angle_std
        mutated[..., 2] += angle_noise * mask

        # Keep angles in [0, 360)
        mutated[..., 2] = mutated[..., 2] % 360.0

        return mutated

    def crossover(
        self,
        configs: torch.Tensor,
        crossover_prob: float = 0.5
    ) -> torch.Tensor:
        """
        Apply crossover between pairs of configurations.

        Swaps tree positions between parent pairs.

        Args:
            configs: (batch, n_trees, 3) configurations
            crossover_prob: Probability of crossing over each tree

        Returns:
            offspring: (batch, n_trees, 3) new configurations
        """
        batch_size = configs.shape[0]
        if batch_size < 2:
            return configs.clone()

        # Shuffle to create random pairs
        perm = torch.randperm(batch_size, device=self.device)
        shuffled = configs[perm]

        # Crossover mask: which trees to swap
        mask = torch.rand(batch_size, self.n_trees, device=self.device) < crossover_prob

        # Create offspring
        offspring = configs.clone()
        offspring[mask] = shuffled[mask]

        return offspring

    def compress_towards_center(
        self,
        configs: torch.Tensor,
        strength: float = 0.1
    ) -> torch.Tensor:
        """
        Move trees towards the center of their configuration.

        Args:
            configs: (batch, n_trees, 3) configurations
            strength: How much to move (0=none, 1=all the way to center)

        Returns:
            compressed: (batch, n_trees, 3)
        """
        compressed = configs.clone()

        # Compute center of each configuration
        centers = configs[..., :2].mean(dim=1, keepdim=True)  # (batch, 1, 2)

        # Move towards center
        delta = centers - configs[..., :2]
        compressed[..., :2] += strength * delta

        return compressed

    def step(
        self,
        elite_ratio: float = 0.25,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        position_std: float = 0.05,
        angle_std: float = 5.0,
        compression_strength: float = 0.0,
    ) -> dict:
        """
        Perform one generation of evolution.

        Returns:
            stats: Dictionary with generation statistics
        """
        self.generation += 1

        # Evaluate current population
        fitness, side_lengths, overlap_counts = self.evaluate()

        # Track best
        best_idx = fitness.argmin()
        best_fit = fitness[best_idx].item()
        best_side = side_lengths[best_idx].item()
        best_overlaps = overlap_counts[best_idx].item()

        if best_fit < self.best_fitness:
            self.best_fitness = best_fit
            self.best_side_length = best_side
            self.best_config = self.population[best_idx].clone()

        # Selection
        elite, elite_indices = self.select_elite(fitness, elite_ratio)
        n_elite = elite.shape[0]

        # Generate offspring through mutation and crossover
        n_offspring = self.pop_size - n_elite

        # Expand elite to create offspring pool
        offspring_pool = elite.repeat((n_offspring // n_elite + 1), 1, 1)[:n_offspring]

        # Apply crossover
        offspring_pool = self.crossover(offspring_pool, crossover_rate)

        # Apply mutation
        offspring_pool = self.mutate(
            offspring_pool,
            position_std=position_std,
            angle_std=angle_std,
            mutation_prob=mutation_rate
        )

        # Optional compression
        if compression_strength > 0:
            offspring_pool = self.compress_towards_center(offspring_pool, compression_strength)

        # Combine elite (unchanged) and offspring
        self.population = torch.cat([elite, offspring_pool], dim=0)

        # Collect statistics
        stats = {
            'generation': self.generation,
            'best_fitness': best_fit,
            'best_side_length': best_side,
            'best_overlaps': int(best_overlaps),
            'mean_fitness': fitness.mean().item(),
            'mean_side_length': side_lengths.mean().item(),
            'mean_overlaps': overlap_counts.float().mean().item(),
            'global_best_fitness': self.best_fitness,
            'global_best_side_length': self.best_side_length,
        }

        self.history.append(stats)
        return stats

    def run(
        self,
        n_generations: int = 100,
        patience: int = 20,
        min_improvement: float = 1e-4,
        verbose: bool = True,
        **step_kwargs
    ) -> torch.Tensor:
        """
        Run evolution for multiple generations.

        Args:
            n_generations: Maximum number of generations
            patience: Stop if no improvement for this many generations
            min_improvement: Minimum improvement to reset patience
            verbose: Print progress

        Returns:
            best_config: (n_trees, 3) best configuration found
        """
        stagnant_generations = 0
        last_best = float('inf')

        start_time = time.perf_counter()

        for gen in range(n_generations):
            stats = self.step(**step_kwargs)

            # Check for improvement
            if last_best - stats['global_best_fitness'] > min_improvement:
                stagnant_generations = 0
                last_best = stats['global_best_fitness']
            else:
                stagnant_generations += 1

            if verbose and gen % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"Gen {gen:4d}: best={stats['best_side_length']:.4f} "
                      f"(overlaps={stats['best_overlaps']}) "
                      f"mean={stats['mean_side_length']:.4f} "
                      f"global_best={stats['global_best_side_length']:.4f} "
                      f"[{elapsed:.1f}s]")

            # Early stopping
            if stagnant_generations >= patience:
                if verbose:
                    print(f"Early stopping at generation {gen} (no improvement for {patience} gens)")
                break

        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"\nCompleted in {elapsed:.1f}s")
            print(f"Best side length: {self.best_side_length:.4f}")

        return self.best_config

    def export_best(self) -> np.ndarray:
        """Export best configuration as numpy array."""
        if self.best_config is None:
            raise ValueError("No best configuration found. Run optimization first.")
        return self.best_config.cpu().numpy()


def load_greedy_from_rust(
    rust_dir: Path,
    n: int,
    max_configs: int = 10
) -> List[np.ndarray]:
    """
    Load greedy solutions from Rust best-of-N runs.

    Args:
        rust_dir: Path to rust directory
        n: Number of trees
        max_configs: Maximum configs to load

    Returns:
        List of (n_trees, 3) numpy arrays
    """
    # Look for solution files or generate on the fly
    # For now, we'll generate random valid-ish configs

    configs = []
    box_size = estimate_box_size(n)

    for i in range(max_configs):
        # Simple grid-based initialization
        config = generate_grid_config(n, box_size)
        configs.append(config)

    return configs


def estimate_box_size(n: int) -> float:
    """Estimate initial box size for n trees based on tree area."""
    tree_area = 0.35  # Approximate tree area
    total_area = n * tree_area
    # Assume 60% packing efficiency initially
    side = np.sqrt(total_area / 0.6)
    return side


def generate_grid_config(n: int, box_size: float) -> np.ndarray:
    """Generate a grid-based initial configuration."""
    # Grid dimensions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    spacing = box_size / max(rows, cols)

    config = np.zeros((n, 3), dtype=np.float32)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            # Center the grid
            x = (col - (cols - 1) / 2) * spacing
            y = (row - (rows - 1) / 2) * spacing
            # Random rotation (0, 45, 90, ...)
            angle = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])

            config[idx] = [x, y, angle]
            idx += 1

    return config


def test_population_opt(n_trees: int = 20, pop_size: int = 32, n_gens: int = 50):
    """Test the population optimizer."""
    print(f"Population Optimizer Test (n={n_trees}, pop={pop_size})")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    opt = PopulationOptimizer(
        n_trees=n_trees,
        pop_size=pop_size,
        device=device,
        overlap_penalty=100.0
    )

    # Initialize from grid configs
    initial_configs = [generate_grid_config(n_trees, 3.0) for _ in range(pop_size)]
    opt.initialize_from_greedy(initial_configs)

    print(f"\nInitial population shape: {opt.population.shape}")

    # Run optimization
    best = opt.run(
        n_generations=n_gens,
        patience=20,
        verbose=True,
        elite_ratio=0.25,
        mutation_rate=0.4,
        crossover_rate=0.3,
        position_std=0.1,
        angle_std=15.0,
        compression_strength=0.02,
    )

    print(f"\nBest configuration shape: {best.shape}")

    # Validate best solution
    print("\nValidating best solution...")
    best_batch = best.unsqueeze(0)
    _, bbox, overlaps, side_lengths = evaluate_configs(opt.tree_tensor, best_batch)
    overlap_count = gpu_count_overlaps(overlaps).item()

    print(f"  Side length: {side_lengths.item():.4f}")
    print(f"  Overlaps: {overlap_count}")

    return opt


def run_full_optimization(n: int = 200, pop_size: int = 64, n_gens: int = 200):
    """Run full optimization for competition."""
    print(f"\n{'='*60}")
    print(f"Full Optimization: n={n}, pop={pop_size}, gens={n_gens}")
    print(f"{'='*60}\n")

    device = get_device()
    print(f"Device: {device}")

    # Estimate box size
    box_size = estimate_box_size(n) * 1.2  # 20% margin

    opt = PopulationOptimizer(
        n_trees=n,
        pop_size=pop_size,
        device=device,
        overlap_penalty=1000.0  # Higher penalty for full problem
    )

    # Initialize from random grid configs
    initial_configs = [generate_grid_config(n, box_size) for _ in range(pop_size)]
    opt.initialize_from_greedy(initial_configs)

    # Run with adaptive parameters
    best = opt.run(
        n_generations=n_gens,
        patience=50,
        verbose=True,
        elite_ratio=0.2,
        mutation_rate=0.3,
        crossover_rate=0.4,
        position_std=0.08,
        angle_std=10.0,
        compression_strength=0.01,
    )

    return opt


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        # Run full optimization
        run_full_optimization(n=200, pop_size=64, n_gens=300)
    else:
        # Quick test
        test_population_opt(n_trees=20, pop_size=32, n_gens=100)
