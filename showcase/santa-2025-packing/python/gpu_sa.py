#!/usr/bin/env python3
"""
Gen107: Simulated Annealing with Hybrid GPU/CPU collision detection.

Based on 70.1 solution analysis:
- 40k SA iterations, 6 restarts per group
- Move types: position (40%), rotation (30%), combined (20%), squeeze (10%)
- Exponential temperature schedule
- Use hybrid collision for accurate overlap detection
"""

import numpy as np
import time
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from hybrid_collision import HybridCollisionChecker
from polygon_collision import transform_trees_batch, polygons_overlap


class MoveType(Enum):
    POSITION = 'position'
    ROTATION = 'rotation'
    COMBINED = 'combined'
    SQUEEZE = 'squeeze'


@dataclass
class SAConfig:
    """Simulated Annealing configuration."""
    iterations: int = 40000
    initial_temp: float = 2.0
    final_temp: float = 0.01
    position_scale: float = 0.1
    angle_scale: float = 15.0  # degrees
    squeeze_factor: float = 0.995
    move_probs: dict = None  # Will be set in __post_init__

    overlap_penalty: float = 1000.0
    restarts: int = 6

    def __post_init__(self):
        if self.move_probs is None:
            self.move_probs = {
                MoveType.POSITION: 0.4,
                MoveType.ROTATION: 0.3,
                MoveType.COMBINED: 0.2,
                MoveType.SQUEEZE: 0.1,
            }


class HybridSA:
    """
    Simulated Annealing optimizer using hybrid GPU/CPU collision.
    """

    def __init__(self, checker: Optional[HybridCollisionChecker] = None):
        self.checker = checker or HybridCollisionChecker()
        self.rng = np.random.default_rng()

    def _choose_move_type(self, config: SAConfig) -> MoveType:
        """Choose move type based on probabilities."""
        r = self.rng.random()
        cumsum = 0.0
        for move_type, prob in config.move_probs.items():
            cumsum += prob
            if r < cumsum:
                return move_type
        return MoveType.POSITION  # Fallback

    def _apply_move(
        self,
        configs: np.ndarray,  # (n_trees, 3)
        move_type: MoveType,
        sa_config: SAConfig,
        temp: float
    ) -> Tuple[np.ndarray, int]:
        """
        Apply a move to the configuration.

        Returns:
            (new_configs, tree_idx) - new config and which tree was moved (-1 for squeeze)
        """
        n_trees = configs.shape[0]
        new_configs = configs.copy()

        # Scale moves by temperature for adaptive step size
        temp_scale = max(0.1, temp / sa_config.initial_temp)

        if move_type == MoveType.POSITION:
            tree_idx = self.rng.integers(0, n_trees)
            delta = self.rng.standard_normal(2) * sa_config.position_scale * temp_scale
            new_configs[tree_idx, :2] += delta
            return new_configs, tree_idx

        elif move_type == MoveType.ROTATION:
            tree_idx = self.rng.integers(0, n_trees)
            delta = self.rng.standard_normal() * sa_config.angle_scale * temp_scale
            new_configs[tree_idx, 2] += delta
            # Keep angle in [0, 360)
            new_configs[tree_idx, 2] = new_configs[tree_idx, 2] % 360
            return new_configs, tree_idx

        elif move_type == MoveType.COMBINED:
            tree_idx = self.rng.integers(0, n_trees)
            delta_pos = self.rng.standard_normal(2) * sa_config.position_scale * temp_scale
            delta_angle = self.rng.standard_normal() * sa_config.angle_scale * temp_scale
            new_configs[tree_idx, :2] += delta_pos
            new_configs[tree_idx, 2] = (new_configs[tree_idx, 2] + delta_angle) % 360
            return new_configs, tree_idx

        elif move_type == MoveType.SQUEEZE:
            # Squeeze all trees toward center
            center = new_configs[:, :2].mean(axis=0)
            new_configs[:, :2] = (
                new_configs[:, :2] * sa_config.squeeze_factor +
                center * (1 - sa_config.squeeze_factor)
            )
            return new_configs, -1

        return new_configs, -1

    def optimize(
        self,
        initial_configs: np.ndarray,  # (n_trees, 3)
        sa_config: Optional[SAConfig] = None,
        verbose: bool = False,
        callback: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Run simulated annealing optimization.

        Args:
            initial_configs: Starting configuration (n_trees, 3)
            sa_config: SA parameters
            verbose: Print progress
            callback: Called after each iteration with (iter, temp, score, best_score)

        Returns:
            (best_configs, best_score, best_overlaps)
        """
        if sa_config is None:
            sa_config = SAConfig()

        n_trees = initial_configs.shape[0]
        configs = initial_configs.copy()

        # Initial evaluation
        n_overlaps, side_length = self.checker.check_config_overlaps(configs)
        current_score = side_length + sa_config.overlap_penalty * n_overlaps

        best_configs = configs.copy()
        best_score = current_score
        best_overlaps = n_overlaps

        # Temperature schedule (exponential decay)
        log_temp_ratio = np.log(sa_config.final_temp / sa_config.initial_temp)

        accepted = 0
        improved = 0

        start_time = time.perf_counter()

        for iteration in range(sa_config.iterations):
            # Current temperature
            progress = iteration / sa_config.iterations
            temp = sa_config.initial_temp * np.exp(log_temp_ratio * progress)

            # Generate move
            move_type = self._choose_move_type(sa_config)
            new_configs, tree_idx = self._apply_move(configs, move_type, sa_config, temp)

            # Evaluate new configuration
            new_overlaps, new_side = self.checker.check_config_overlaps(new_configs)
            new_score = new_side + sa_config.overlap_penalty * new_overlaps

            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0 or self.rng.random() < np.exp(-delta / temp):
                configs = new_configs
                current_score = new_score
                n_overlaps = new_overlaps
                accepted += 1

                if new_score < best_score:
                    best_configs = configs.copy()
                    best_score = new_score
                    best_overlaps = new_overlaps
                    improved += 1

            if callback:
                callback(iteration, temp, current_score, best_score)

            if verbose and (iteration + 1) % 5000 == 0:
                elapsed = time.perf_counter() - start_time
                rate = (iteration + 1) / elapsed
                print(f"  Iter {iteration+1:5d}: temp={temp:.4f}, "
                      f"score={current_score:.2f}, best={best_score:.2f}, "
                      f"overlaps={n_overlaps}, rate={rate:.0f}/sec")

        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"  Done: {sa_config.iterations} iters in {elapsed:.1f}s "
                  f"({sa_config.iterations/elapsed:.0f}/sec)")
            print(f"  Accepted: {accepted} ({100*accepted/sa_config.iterations:.1f}%)")
            print(f"  Improved: {improved}")

        return best_configs, best_score, best_overlaps

    def optimize_with_restarts(
        self,
        initial_configs: np.ndarray,  # (n_trees, 3)
        sa_config: Optional[SAConfig] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Run SA with multiple restarts, keeping best result.
        """
        if sa_config is None:
            sa_config = SAConfig()

        best_configs = initial_configs.copy()
        best_score = float('inf')
        best_overlaps = -1

        for restart in range(sa_config.restarts):
            if verbose:
                print(f"\nRestart {restart + 1}/{sa_config.restarts}")

            # Slightly perturb initial config for each restart (except first)
            if restart > 0:
                perturbed = initial_configs.copy()
                perturbed[:, :2] += self.rng.standard_normal((initial_configs.shape[0], 2)) * 0.05
                perturbed[:, 2] += self.rng.standard_normal(initial_configs.shape[0]) * 5
                perturbed[:, 2] = perturbed[:, 2] % 360
                start_configs = perturbed
            else:
                start_configs = initial_configs

            configs, score, overlaps = self.optimize(
                start_configs, sa_config, verbose=verbose
            )

            if score < best_score:
                best_configs = configs
                best_score = score
                best_overlaps = overlaps
                if verbose:
                    print(f"  New best! Score: {best_score:.2f}")

        return best_configs, best_score, best_overlaps


def create_initial_config(n_trees: int, box_size: float = 10.0) -> np.ndarray:
    """Create a simple initial configuration with grid placement."""
    configs = np.zeros((n_trees, 3), dtype=np.float64)

    # Grid arrangement
    side = int(np.ceil(np.sqrt(n_trees)))
    spacing = box_size / side

    for i in range(n_trees):
        row = i // side
        col = i % side
        configs[i, 0] = (col - side / 2 + 0.5) * spacing
        configs[i, 1] = (row - side / 2 + 0.5) * spacing
        configs[i, 2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])

    return configs


def benchmark_sa():
    """Benchmark SA performance."""
    print("SA Benchmark")
    print("=" * 50)

    sa = HybridSA()

    for n_trees in [20, 50]:
        print(f"\n--- n_trees = {n_trees} ---")

        initial_configs = create_initial_config(n_trees)

        # Quick test with few iterations
        sa_config = SAConfig(
            iterations=1000,
            restarts=1,
        )

        start = time.perf_counter()
        best_configs, best_score, best_overlaps = sa.optimize(
            initial_configs, sa_config, verbose=True
        )
        elapsed = time.perf_counter() - start

        print(f"  Final: score={best_score:.2f}, overlaps={best_overlaps}")
        print(f"  Time: {elapsed:.1f}s")

        # Verify final configuration
        final_overlaps, final_side = sa.checker.check_config_overlaps(best_configs)
        print(f"  Verify: side={final_side:.2f}, overlaps={final_overlaps}")


def test_full_optimization():
    """Test full optimization pipeline for small n."""
    print("\nFull Optimization Test (n=20)")
    print("=" * 50)

    sa = HybridSA()
    n_trees = 20

    initial_configs = create_initial_config(n_trees, box_size=8.0)

    # Check initial state
    init_overlaps, init_side = sa.checker.check_config_overlaps(initial_configs)
    print(f"Initial: side={init_side:.2f}, overlaps={init_overlaps}")

    # Full optimization
    sa_config = SAConfig(
        iterations=10000,
        restarts=3,
        initial_temp=2.0,
        final_temp=0.01,
    )

    start = time.perf_counter()
    best_configs, best_score, best_overlaps = sa.optimize_with_restarts(
        initial_configs, sa_config, verbose=True
    )
    elapsed = time.perf_counter() - start

    print(f"\nFinal result:")
    print(f"  Score: {best_score:.2f}")
    print(f"  Side length: {best_score - 1000 * best_overlaps:.2f}")
    print(f"  Overlaps: {best_overlaps}")
    print(f"  Time: {elapsed:.1f}s")

    # Validate with full overlap check
    vertices = transform_trees_batch(best_configs)
    true_overlaps = 0
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            if polygons_overlap(vertices[i], vertices[j]):
                true_overlaps += 1
    print(f"  Verified overlaps (with safety margin): {true_overlaps}")


if __name__ == '__main__':
    benchmark_sa()
    test_full_optimization()
