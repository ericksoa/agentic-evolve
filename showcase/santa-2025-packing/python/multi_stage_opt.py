#!/usr/bin/env python3
"""
Gen107: Multi-Stage Optimization Pipeline.

Based on 70.1 solution architecture:
1. Initial placement (greedy or Rust import)
2. Global rotation optimization
3. Compaction passes
4. Squeeze toward center
5. Simulated annealing
6. Final polish

This pipeline combines GPU-accelerated operations with
accurate CPU polygon collision detection.
"""

import numpy as np
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from hybrid_collision import HybridCollisionChecker
from polygon_collision import transform_trees_batch, polygons_overlap
from gpu_sa import HybridSA, SAConfig, create_initial_config


@dataclass
class PipelineConfig:
    """Configuration for multi-stage optimization."""
    # Global rotation
    rotation_angles: int = 36  # Test every 10 degrees
    rotation_enabled: bool = True

    # Compaction
    compaction_passes: int = 15
    compaction_enabled: bool = True

    # Squeeze
    squeeze_factor: float = 0.99
    squeeze_passes: int = 5
    squeeze_enabled: bool = True

    # SA
    sa_iterations: int = 40000
    sa_restarts: int = 6
    sa_enabled: bool = True

    # Polish
    polish_iterations: int = 5000
    polish_enabled: bool = True


class MultiStageOptimizer:
    """
    Multi-stage optimization pipeline for tree packing.
    """

    def __init__(self, checker: Optional[HybridCollisionChecker] = None):
        self.checker = checker or HybridCollisionChecker()
        self.sa = HybridSA(self.checker)

    def global_rotation_optimize(
        self,
        configs: np.ndarray,
        n_angles: int = 36,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Find the best global rotation angle that minimizes side length.

        Rotates ALL trees together around the center point.
        """
        n_trees = configs.shape[0]
        center = configs[:, :2].mean(axis=0)

        best_configs = configs.copy()
        best_score = float('inf')
        best_overlaps = -1
        best_angle = 0

        for i in range(n_angles):
            angle = i * 360.0 / n_angles
            angle_rad = np.radians(angle)

            # Rotate all positions around center
            rotated = configs.copy()
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            for j in range(n_trees):
                dx = configs[j, 0] - center[0]
                dy = configs[j, 1] - center[1]
                rotated[j, 0] = center[0] + dx * cos_a - dy * sin_a
                rotated[j, 1] = center[1] + dx * sin_a + dy * cos_a
                rotated[j, 2] = (configs[j, 2] + angle) % 360

            # Evaluate
            n_overlaps, side_length = self.checker.check_config_overlaps(rotated)

            # Only accept if no overlaps created
            if n_overlaps == 0:
                if side_length < best_score:
                    best_score = side_length
                    best_configs = rotated.copy()
                    best_overlaps = n_overlaps
                    best_angle = angle

        if verbose:
            print(f"  Global rotation: best angle={best_angle:.1f}°, "
                  f"side={best_score:.3f}, overlaps={best_overlaps}")

        return best_configs

    def compaction_pass(
        self,
        configs: np.ndarray,
        direction: str = 'center',
        step_size: float = 0.01,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Move each tree toward a target, checking for overlaps.

        direction: 'center', 'left', 'right', 'up', 'down'
        """
        n_trees = configs.shape[0]
        result = configs.copy()

        center = configs[:, :2].mean(axis=0)

        # Order trees by distance from target (farthest first)
        if direction == 'center':
            distances = np.linalg.norm(configs[:, :2] - center, axis=1)
            target = center
        elif direction == 'left':
            distances = configs[:, 0]  # x coordinate
            target = np.array([configs[:, 0].min() - 1, center[1]])
        elif direction == 'right':
            distances = -configs[:, 0]
            target = np.array([configs[:, 0].max() + 1, center[1]])
        elif direction == 'up':
            distances = -configs[:, 1]
            target = np.array([center[0], configs[:, 1].max() + 1])
        elif direction == 'down':
            distances = configs[:, 1]
            target = np.array([center[0], configs[:, 1].min() - 1])
        else:
            return result

        order = np.argsort(-distances)  # Farthest first

        moved = 0
        for idx in order:
            # Try to move toward target
            direction_vec = target - result[idx, :2]
            dist = np.linalg.norm(direction_vec)
            if dist < 0.01:
                continue

            direction_vec = direction_vec / dist * step_size

            # Try move
            new_configs = result.copy()
            new_configs[idx, :2] += direction_vec

            # Check if move creates overlap
            n_overlaps, _ = self.checker.check_config_overlaps(new_configs)
            if n_overlaps == 0:
                result = new_configs
                moved += 1

        if verbose:
            print(f"    Compaction ({direction}): moved {moved}/{n_trees} trees")

        return result

    def squeeze_toward_center(
        self,
        configs: np.ndarray,
        factor: float = 0.99,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Squeeze all positions toward center by a factor.
        Only accept if no overlaps created.
        """
        center = configs[:, :2].mean(axis=0)

        squeezed = configs.copy()
        squeezed[:, :2] = configs[:, :2] * factor + center * (1 - factor)

        # Check for overlaps
        n_overlaps, side_length = self.checker.check_config_overlaps(squeezed)

        if n_overlaps == 0:
            if verbose:
                print(f"  Squeeze: accepted, side={side_length:.3f}")
            return squeezed
        else:
            if verbose:
                print(f"  Squeeze: rejected (would create {n_overlaps} overlaps)")
            return configs

    def polish(
        self,
        configs: np.ndarray,
        iterations: int = 5000,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Final polish with tight SA parameters.
        """
        sa_config = SAConfig(
            iterations=iterations,
            initial_temp=0.5,
            final_temp=0.001,
            position_scale=0.02,
            angle_scale=5.0,
            restarts=1,
        )

        result, score, overlaps = self.sa.optimize(
            configs, sa_config, verbose=verbose
        )

        return result

    def optimize(
        self,
        initial_configs: np.ndarray,
        pipeline_config: Optional[PipelineConfig] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Run full multi-stage optimization pipeline.

        Args:
            initial_configs: Starting configuration (n_trees, 3)
            pipeline_config: Pipeline parameters
            verbose: Print progress

        Returns:
            (best_configs, best_side_length, n_overlaps)
        """
        if pipeline_config is None:
            pipeline_config = PipelineConfig()

        configs = initial_configs.copy()
        n_trees = configs.shape[0]

        # Initial evaluation
        n_overlaps, side_length = self.checker.check_config_overlaps(configs)
        if verbose:
            print(f"Initial: side={side_length:.3f}, overlaps={n_overlaps}")

        # Stage 1: Global rotation
        if pipeline_config.rotation_enabled:
            if verbose:
                print("\nStage 1: Global rotation optimization")
            configs = self.global_rotation_optimize(
                configs, pipeline_config.rotation_angles, verbose
            )

        # Stage 2: Compaction passes
        if pipeline_config.compaction_enabled:
            if verbose:
                print(f"\nStage 2: Compaction ({pipeline_config.compaction_passes} passes)")
            directions = ['center', 'left', 'right', 'up', 'down']
            for p in range(pipeline_config.compaction_passes):
                direction = directions[p % len(directions)]
                configs = self.compaction_pass(
                    configs, direction, step_size=0.02, verbose=verbose
                )

        # Stage 3: Squeeze
        if pipeline_config.squeeze_enabled:
            if verbose:
                print(f"\nStage 3: Squeeze ({pipeline_config.squeeze_passes} passes)")
            for _ in range(pipeline_config.squeeze_passes):
                configs = self.squeeze_toward_center(
                    configs, pipeline_config.squeeze_factor, verbose=False
                )
            n_overlaps, side_length = self.checker.check_config_overlaps(configs)
            if verbose:
                print(f"  After squeeze: side={side_length:.3f}, overlaps={n_overlaps}")

        # Stage 4: Simulated annealing
        if pipeline_config.sa_enabled:
            if verbose:
                print(f"\nStage 4: Simulated annealing "
                      f"({pipeline_config.sa_iterations} iters x {pipeline_config.sa_restarts} restarts)")

            sa_config = SAConfig(
                iterations=pipeline_config.sa_iterations,
                restarts=pipeline_config.sa_restarts,
                initial_temp=2.0,
                final_temp=0.01,
            )

            configs, score, n_overlaps = self.sa.optimize_with_restarts(
                configs, sa_config, verbose=verbose
            )
            side_length = score if n_overlaps == 0 else score - 1000 * n_overlaps

        # Stage 5: Final polish
        if pipeline_config.polish_enabled:
            if verbose:
                print(f"\nStage 5: Final polish ({pipeline_config.polish_iterations} iters)")
            configs = self.polish(configs, pipeline_config.polish_iterations, verbose=verbose)

        # Final evaluation
        n_overlaps, side_length = self.checker.check_config_overlaps(configs)
        if verbose:
            print(f"\nFinal: side={side_length:.3f}, overlaps={n_overlaps}")

        return configs, side_length, n_overlaps


def load_rust_packing(n: int, rust_dir: Path) -> Optional[np.ndarray]:
    """
    Load a packing from Rust greedy algorithm output.

    This allows using Rust's fast greedy as initial placement.
    """
    # Run Rust to generate packing
    # TODO: Implement Rust JSON output for single n
    return None


def benchmark_pipeline():
    """Benchmark the multi-stage pipeline."""
    print("Multi-Stage Pipeline Benchmark")
    print("=" * 60)

    optimizer = MultiStageOptimizer()

    for n_trees in [20, 50]:
        print(f"\n{'='*60}")
        print(f"n_trees = {n_trees}")
        print("=" * 60)

        # Create initial config
        initial = create_initial_config(n_trees, box_size=n_trees * 0.4)

        # Quick pipeline for testing
        config = PipelineConfig(
            rotation_angles=12,
            compaction_passes=5,
            squeeze_passes=3,
            sa_iterations=5000,
            sa_restarts=2,
            polish_iterations=2000,
        )

        start = time.perf_counter()
        final_configs, side_length, overlaps = optimizer.optimize(
            initial, config, verbose=True
        )
        elapsed = time.perf_counter() - start

        print(f"\nTotal time: {elapsed:.1f}s")
        print(f"Result: side={side_length:.3f}, overlaps={overlaps}")


def compare_with_rust():
    """
    Compare Python pipeline with Rust evolved algorithm.

    This requires running Rust benchmark and comparing results.
    """
    print("\nComparison with Rust (n=20)")
    print("=" * 60)

    # Run Python pipeline
    optimizer = MultiStageOptimizer()

    n_trees = 20
    initial = create_initial_config(n_trees, box_size=8.0)

    config = PipelineConfig(
        sa_iterations=10000,
        sa_restarts=3,
    )

    start = time.perf_counter()
    final_configs, side_length, overlaps = optimizer.optimize(
        initial, config, verbose=True
    )
    python_time = time.perf_counter() - start

    print(f"\nPython result:")
    print(f"  Side length: {side_length:.3f}")
    print(f"  Overlaps: {overlaps}")
    print(f"  Time: {python_time:.1f}s")

    # TODO: Compare with Rust
    # rust_side = run_rust_benchmark(n_trees)
    # print(f"\nRust result: {rust_side:.3f}")


if __name__ == '__main__':
    benchmark_pipeline()
