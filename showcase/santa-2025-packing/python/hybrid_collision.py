#!/usr/bin/env python3
"""
Gen107: Hybrid GPU/CPU collision detection.

Key insight from 70.1 solution:
- GPU bbox as FILTER (not collision detection)
- CPU polygon check only for bbox-overlapping pairs (~10% of pairs)
- This gives best of both worlds: GPU parallelism + accurate collision

Architecture:
1. GPU: Transform all trees in batch (fast)
2. GPU: Compute pairwise bbox overlaps (fast)
3. CPU: Polygon collision for bbox candidates only (accurate)
"""

import torch
import numpy as np
import time
from typing import Tuple, Optional

# Import GPU primitives from Gen106
from gpu_primitives import (
    TreeTensor, get_device,
    gpu_transform_trees, gpu_compute_bbox, gpu_check_bbox_overlaps,
    gpu_score_configs
)

# Import Numba collision from Gen107
from polygon_collision import (
    polygons_overlap_fast, transform_trees_batch, transform_tree
)


class HybridCollisionChecker:
    """
    Hybrid GPU/CPU collision detection.

    Uses GPU for fast bbox filtering, CPU for accurate polygon collision.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.tree_tensor = TreeTensor(self.device)

        # Warmup Numba JIT
        self._warmup_numba()

    def _warmup_numba(self):
        """Warmup Numba JIT compilation."""
        warmup_configs = np.zeros((50, 3), dtype=np.float64)
        warmup_configs[:, 0] = np.linspace(-5, 5, 50)
        warmup_verts = transform_trees_batch(warmup_configs)
        _ = polygons_overlap_fast(warmup_verts[0], warmup_verts[1])

    def check_config_overlaps(
        self,
        configs: np.ndarray,  # (n_trees, 3) - [x, y, angle_deg]
    ) -> Tuple[int, float]:
        """
        Check overlaps in a single configuration using hybrid approach.

        Args:
            configs: (n_trees, 3) array of tree configurations

        Returns:
            (n_overlaps, side_length) tuple
        """
        n_trees = configs.shape[0]

        # 1. GPU: Transform and get bbox overlaps
        configs_t = torch.tensor(configs, dtype=torch.float32, device=self.device)
        configs_t = configs_t.unsqueeze(0)  # (1, n_trees, 3)

        transformed = gpu_transform_trees(self.tree_tensor.base_vertices, configs_t)
        bbox = gpu_compute_bbox(transformed)
        bbox_overlaps = gpu_check_bbox_overlaps(bbox)
        side_lengths = gpu_score_configs(bbox)

        side_length = side_lengths[0].item()

        # 2. Get candidate pairs from GPU
        # bbox_overlaps[0] is (n_trees, n_trees) boolean tensor
        # We only want upper triangle
        candidates = []
        bbox_matrix = bbox_overlaps[0].cpu().numpy()
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                if bbox_matrix[i, j]:
                    candidates.append((i, j))

        if not candidates:
            return 0, side_length

        # 3. CPU: Transform vertices (more precise float64)
        vertices = transform_trees_batch(configs)

        # 4. CPU: Check polygon collision for candidates only
        n_overlaps = 0
        for i, j in candidates:
            if polygons_overlap_fast(vertices[i], vertices[j]):
                n_overlaps += 1

        return n_overlaps, side_length

    def check_batch_overlaps(
        self,
        batch_configs: np.ndarray,  # (batch, n_trees, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check overlaps for a batch of configurations.

        Args:
            batch_configs: (batch, n_trees, 3) array of configurations

        Returns:
            (overlap_counts, side_lengths) arrays of shape (batch,)
        """
        batch_size = batch_configs.shape[0]
        n_trees = batch_configs.shape[1]

        # 1. GPU: Batch transform and bbox
        configs_t = torch.tensor(batch_configs, dtype=torch.float32, device=self.device)

        transformed = gpu_transform_trees(self.tree_tensor.base_vertices, configs_t)
        bbox = gpu_compute_bbox(transformed)
        bbox_overlaps = gpu_check_bbox_overlaps(bbox)
        side_lengths = gpu_score_configs(bbox)

        side_lengths_np = side_lengths.cpu().numpy()

        # 2. For each config, check polygon overlaps
        overlap_counts = np.zeros(batch_size, dtype=np.int32)
        bbox_overlaps_np = bbox_overlaps.cpu().numpy()

        for b in range(batch_size):
            # Get candidate pairs
            candidates = []
            for i in range(n_trees):
                for j in range(i + 1, n_trees):
                    if bbox_overlaps_np[b, i, j]:
                        candidates.append((i, j))

            if not candidates:
                continue

            # Transform vertices (CPU, float64)
            vertices = transform_trees_batch(batch_configs[b])

            # Check polygon collision
            for i, j in candidates:
                if polygons_overlap_fast(vertices[i], vertices[j]):
                    overlap_counts[b] += 1

        return overlap_counts, side_lengths_np

    def evaluate_config(
        self,
        configs: np.ndarray,  # (n_trees, 3)
        overlap_penalty: float = 1000.0
    ) -> float:
        """
        Evaluate a single configuration, returning fitness score.

        Lower is better: side_length + penalty * n_overlaps
        """
        n_overlaps, side_length = self.check_config_overlaps(configs)
        return side_length + overlap_penalty * n_overlaps

    def is_valid(
        self,
        configs: np.ndarray,  # (n_trees, 3)
    ) -> bool:
        """Check if configuration has no overlaps."""
        n_overlaps, _ = self.check_config_overlaps(configs)
        return n_overlaps == 0


def benchmark_hybrid():
    """Benchmark hybrid collision detection."""
    print("Hybrid Collision Benchmark")
    print("=" * 50)

    checker = HybridCollisionChecker()
    print(f"Device: {checker.device}")

    # Generate test configurations
    np.random.seed(42)

    for n_trees in [20, 50, 100, 200]:
        print(f"\n--- n_trees = {n_trees} ---")

        # Random configs (will have many overlaps)
        configs = np.zeros((n_trees, 3), dtype=np.float64)
        configs[:, 0] = np.random.uniform(-5, 5, n_trees)
        configs[:, 1] = np.random.uniform(-5, 5, n_trees)
        configs[:, 2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315], n_trees)

        # Warmup
        for _ in range(3):
            checker.check_config_overlaps(configs)

        # Sync GPU
        if checker.device.type == 'mps':
            torch.mps.synchronize()
        elif checker.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        n_iters = 20
        start = time.perf_counter()
        for _ in range(n_iters):
            n_overlaps, side_length = checker.check_config_overlaps(configs)

        if checker.device.type == 'mps':
            torch.mps.synchronize()
        elif checker.device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        time_per_check = elapsed / n_iters * 1000

        n_pairs = n_trees * (n_trees - 1) // 2

        print(f"  Time: {time_per_check:.2f} ms/check")
        print(f"  Pairs: {n_pairs}")
        print(f"  Overlaps: {n_overlaps}")
        print(f"  Side length: {side_length:.2f}")
        print(f"  Throughput: {n_iters/elapsed:.0f} configs/sec")

    # Batch benchmark
    print("\n--- Batch benchmark (n=200, batch=32) ---")
    n_trees = 200
    batch_size = 32

    batch_configs = np.zeros((batch_size, n_trees, 3), dtype=np.float64)
    for b in range(batch_size):
        batch_configs[b, :, 0] = np.random.uniform(-5, 5, n_trees)
        batch_configs[b, :, 1] = np.random.uniform(-5, 5, n_trees)
        batch_configs[b, :, 2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315], n_trees)

    # Warmup
    checker.check_batch_overlaps(batch_configs)

    if checker.device.type == 'mps':
        torch.mps.synchronize()

    n_iters = 5
    start = time.perf_counter()
    for _ in range(n_iters):
        overlap_counts, side_lengths = checker.check_batch_overlaps(batch_configs)

    if checker.device.type == 'mps':
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    time_per_batch = elapsed / n_iters * 1000
    time_per_config = time_per_batch / batch_size

    print(f"  Batch time: {time_per_batch:.1f} ms")
    print(f"  Per config: {time_per_config:.2f} ms")
    print(f"  Throughput: {batch_size * n_iters / elapsed:.0f} configs/sec")
    print(f"  Avg overlaps: {overlap_counts.mean():.1f}")


def compare_gpu_only_vs_hybrid():
    """
    Compare GPU-only (bbox) vs hybrid (bbox+polygon) approach.

    This demonstrates why hybrid is necessary - bbox alone is too conservative.
    """
    print("\nGPU-only vs Hybrid Comparison")
    print("=" * 50)

    from gpu_primitives import gpu_count_overlaps

    checker = HybridCollisionChecker()

    # Generate a well-spaced configuration (should have few/no overlaps)
    n_trees = 50

    # Grid arrangement - should have no polygon overlaps but some bbox overlaps
    configs = np.zeros((n_trees, 3), dtype=np.float64)
    side = int(np.ceil(np.sqrt(n_trees)))
    for i in range(n_trees):
        row = i // side
        col = i % side
        configs[i, 0] = col * 0.8 - (side - 1) * 0.4  # x
        configs[i, 1] = row * 1.0 - (side - 1) * 0.5  # y
        configs[i, 2] = 0.0 if (row + col) % 2 == 0 else 45.0  # alternating rotation

    # GPU-only: count bbox overlaps
    configs_t = torch.tensor(configs, dtype=torch.float32, device=checker.device)
    configs_t = configs_t.unsqueeze(0)

    transformed = gpu_transform_trees(checker.tree_tensor.base_vertices, configs_t)
    bbox = gpu_compute_bbox(transformed)
    bbox_overlaps = gpu_check_bbox_overlaps(bbox)
    bbox_count = gpu_count_overlaps(bbox_overlaps)[0].item()

    # Hybrid: count actual polygon overlaps
    polygon_count, side_length = checker.check_config_overlaps(configs)

    print(f"Configuration: {n_trees} trees in grid pattern")
    print(f"Side length: {side_length:.2f}")
    print(f"BBox overlaps (GPU-only): {bbox_count}")
    print(f"Polygon overlaps (hybrid): {polygon_count}")
    print(f"False positives filtered: {bbox_count - polygon_count}")

    if bbox_count > 0:
        filter_ratio = (bbox_count - polygon_count) / bbox_count * 100
        print(f"Filter efficiency: {filter_ratio:.1f}% of bbox overlaps are false positives")


if __name__ == '__main__':
    benchmark_hybrid()
    compare_gpu_only_vs_hybrid()
