#!/usr/bin/env python3
"""
GPU Primitives for Santa 2025 Packing - Gen106

Uses PyTorch with MPS (Metal Performance Shaders) on Apple Silicon
for batched operations on tree configurations.

Hardware target: Apple M2 Pro (19 GPU cores)
"""

import torch
import numpy as np
from typing import Optional, Tuple
import time

# Tree shape constants (15 vertices)
TREE_VERTICES_NP = np.array([
    [0.0, 0.8],       # Tip
    [0.125, 0.5],     # Right top tier
    [0.0625, 0.5],
    [0.2, 0.25],      # Right mid tier
    [0.1, 0.25],
    [0.35, 0.0],      # Right bottom tier
    [0.075, 0.0],     # Right trunk
    [0.075, -0.2],
    [-0.075, -0.2],   # Left trunk
    [-0.075, 0.0],
    [-0.35, 0.0],     # Left bottom tier
    [-0.1, 0.25],     # Left mid tier
    [-0.2, 0.25],
    [-0.0625, 0.5],   # Left top tier
    [-0.125, 0.5],
], dtype=np.float32)


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TreeTensor:
    """
    Batched tree configurations as PyTorch tensors.

    A configuration is a set of N trees, each with (x, y, angle_deg).
    We store B configurations (batch) x N trees x 3 values.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        # Base tree vertices: (15, 2)
        self.base_vertices = torch.tensor(
            TREE_VERTICES_NP,
            dtype=torch.float32,
            device=self.device
        )

    def from_numpy(self, configs: np.ndarray) -> torch.Tensor:
        """
        Convert numpy configurations to tensor.

        Args:
            configs: (batch, n_trees, 3) - each tree has [x, y, angle_deg]

        Returns:
            Tensor of same shape on device
        """
        return torch.tensor(configs, dtype=torch.float32, device=self.device)

    def random_configs(
        self,
        batch_size: int,
        n_trees: int,
        box_size: float = 5.0,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate random tree configurations.

        Args:
            batch_size: Number of configurations
            n_trees: Trees per configuration
            box_size: Maximum coordinate value
            seed: Random seed for reproducibility

        Returns:
            Tensor of shape (batch_size, n_trees, 3)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Random positions in [-box_size/2, box_size/2]
        positions = (torch.rand(batch_size, n_trees, 2, device=self.device) - 0.5) * box_size

        # Random rotations in [0, 360) degrees
        angles = torch.rand(batch_size, n_trees, 1, device=self.device) * 360.0

        return torch.cat([positions, angles], dim=-1)


def gpu_transform_trees(
    base_vertices: torch.Tensor,
    configs: torch.Tensor
) -> torch.Tensor:
    """
    Transform tree vertices by rotation and translation.

    This is the core operation - transform all trees in all configurations
    in parallel on the GPU.

    Args:
        base_vertices: (15, 2) - the base tree polygon
        configs: (batch, n_trees, 3) - [x, y, angle_deg] per tree

    Returns:
        transformed: (batch, n_trees, 15, 2) - all tree vertices
    """
    batch_size, n_trees, _ = configs.shape
    n_vertices = base_vertices.shape[0]

    # Extract positions and angles
    positions = configs[..., :2]  # (batch, n_trees, 2)
    angles_deg = configs[..., 2]  # (batch, n_trees)

    # Convert to radians
    angles_rad = angles_deg * (torch.pi / 180.0)

    # Compute rotation matrices: (batch, n_trees, 2, 2)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)

    # Build rotation matrix
    rot_matrix = torch.stack([
        torch.stack([cos_a, -sin_a], dim=-1),
        torch.stack([sin_a, cos_a], dim=-1)
    ], dim=-2)  # (batch, n_trees, 2, 2)

    # Apply rotation: (batch, n_trees, 15, 2)
    # For row vectors: rotated = vertices @ R^T
    # R = [[cos, -sin], [sin, cos]], R^T = [[cos, sin], [-sin, cos]]
    # rotated[b,n,v,j] = sum_i vertices[v,i] * R^T[i,j]
    #                  = sum_i vertices[v,i] * R[j,i]
    # As einsum: 'vi,bnji->bnvj'
    rotated = torch.einsum('vi,bnji->bnvj', base_vertices, rot_matrix)

    # Add translation: (batch, n_trees, 1, 2) + (batch, n_trees, 15, 2)
    positions_expanded = positions.unsqueeze(2)  # (batch, n_trees, 1, 2)
    transformed = rotated + positions_expanded

    return transformed


def gpu_compute_bbox(transformed_vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute bounding boxes for all trees in all configurations.

    Args:
        transformed_vertices: (batch, n_trees, 15, 2) - tree vertices

    Returns:
        bbox: (batch, n_trees, 4) - [min_x, min_y, max_x, max_y]
    """
    # Min and max over the vertex dimension
    min_coords = transformed_vertices.min(dim=2).values  # (batch, n_trees, 2)
    max_coords = transformed_vertices.max(dim=2).values  # (batch, n_trees, 2)

    # Concatenate: [min_x, min_y, max_x, max_y]
    bbox = torch.cat([min_coords, max_coords], dim=-1)  # (batch, n_trees, 4)

    return bbox


def gpu_check_bbox_overlaps(bbox: torch.Tensor) -> torch.Tensor:
    """
    Check pairwise bounding box overlaps for all trees in all configurations.

    This is a fast filter - if bboxes don't overlap, polygons can't overlap.

    Args:
        bbox: (batch, n_trees, 4) - [min_x, min_y, max_x, max_y]

    Returns:
        overlaps: (batch, n_trees, n_trees) - boolean overlap matrix
                  overlaps[b,i,j] = True if tree i and j bbox overlap in config b
    """
    batch_size, n_trees, _ = bbox.shape

    # Extract components
    min_x = bbox[..., 0]  # (batch, n_trees)
    min_y = bbox[..., 1]
    max_x = bbox[..., 2]
    max_y = bbox[..., 3]

    # Expand for pairwise comparison
    # For overlap: max_x1 > min_x2 AND min_x1 < max_x2 (and same for y)

    # Shape: (batch, n_trees, 1) vs (batch, 1, n_trees)
    min_x_1 = min_x.unsqueeze(2)  # (batch, n_trees, 1)
    min_x_2 = min_x.unsqueeze(1)  # (batch, 1, n_trees)
    max_x_1 = max_x.unsqueeze(2)
    max_x_2 = max_x.unsqueeze(1)
    min_y_1 = min_y.unsqueeze(2)
    min_y_2 = min_y.unsqueeze(1)
    max_y_1 = max_y.unsqueeze(2)
    max_y_2 = max_y.unsqueeze(1)

    # AABB overlap test
    x_overlap = (max_x_1 > min_x_2) & (min_x_1 < max_x_2)
    y_overlap = (max_y_1 > min_y_2) & (min_y_1 < max_y_2)

    overlaps = x_overlap & y_overlap  # (batch, n_trees, n_trees)

    # Zero out diagonal (tree doesn't overlap with itself)
    eye_mask = torch.eye(n_trees, dtype=torch.bool, device=bbox.device)
    eye_mask = eye_mask.unsqueeze(0).expand(batch_size, -1, -1)
    overlaps = overlaps & ~eye_mask

    return overlaps


def gpu_score_configs(bbox: torch.Tensor) -> torch.Tensor:
    """
    Compute the side_length (max of width, height) for each configuration.

    The competition score is sum(side_length^2 / n) over all groups.
    For a single n, minimizing side_length minimizes score.

    Args:
        bbox: (batch, n_trees, 4) - [min_x, min_y, max_x, max_y]

    Returns:
        side_lengths: (batch,) - side length for each configuration
    """
    # Global min/max across all trees in each config
    min_coords = bbox[..., :2].min(dim=1).values  # (batch, 2)
    max_coords = bbox[..., 2:].max(dim=1).values  # (batch, 2)

    # Dimensions
    widths = max_coords[:, 0] - min_coords[:, 0]   # (batch,)
    heights = max_coords[:, 1] - min_coords[:, 1]  # (batch,)

    # Side length is max of width and height
    side_lengths = torch.maximum(widths, heights)

    return side_lengths


def gpu_count_overlaps(overlaps: torch.Tensor) -> torch.Tensor:
    """
    Count the number of overlapping pairs in each configuration.

    Args:
        overlaps: (batch, n_trees, n_trees) - boolean overlap matrix

    Returns:
        counts: (batch,) - number of overlapping pairs per config
    """
    # Sum upper triangle (avoid double counting)
    triu_mask = torch.triu(torch.ones_like(overlaps[0]), diagonal=1)
    triu_mask = triu_mask.unsqueeze(0).expand_as(overlaps)

    return (overlaps & triu_mask).sum(dim=(1, 2))


def gpu_fitness(
    bbox: torch.Tensor,
    overlaps: torch.Tensor,
    overlap_penalty: float = 1000.0
) -> torch.Tensor:
    """
    Compute fitness score for each configuration.

    Lower is better. Score = side_length + penalty * n_overlaps

    Args:
        bbox: (batch, n_trees, 4)
        overlaps: (batch, n_trees, n_trees)
        overlap_penalty: Penalty per overlapping pair

    Returns:
        fitness: (batch,) - fitness scores (lower is better)
    """
    side_lengths = gpu_score_configs(bbox)
    overlap_counts = gpu_count_overlaps(overlaps)

    return side_lengths + overlap_penalty * overlap_counts.float()


# Higher-level convenience function
def evaluate_configs(
    tree_tensor: TreeTensor,
    configs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full evaluation pipeline for configurations.

    Args:
        tree_tensor: TreeTensor instance with base vertices
        configs: (batch, n_trees, 3) tensor of configurations

    Returns:
        transformed: (batch, n_trees, 15, 2) - all vertices
        bbox: (batch, n_trees, 4) - bounding boxes
        overlaps: (batch, n_trees, n_trees) - overlap matrix
        side_lengths: (batch,) - side lengths
    """
    transformed = gpu_transform_trees(tree_tensor.base_vertices, configs)
    bbox = gpu_compute_bbox(transformed)
    overlaps = gpu_check_bbox_overlaps(bbox)
    side_lengths = gpu_score_configs(bbox)

    return transformed, bbox, overlaps, side_lengths


def benchmark_gpu_primitives(
    batch_size: int = 32,
    n_trees: int = 200,
    n_iterations: int = 100
):
    """
    Benchmark GPU primitives performance.
    """
    print(f"GPU Primitives Benchmark")
    print(f"========================")

    device = get_device()
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Trees per config: {n_trees}")
    print(f"Iterations: {n_iterations}")
    print()

    tree_tensor = TreeTensor(device)
    configs = tree_tensor.random_configs(batch_size, n_trees, box_size=10.0)

    # Warmup
    for _ in range(5):
        transformed, bbox, overlaps, side_lengths = evaluate_configs(tree_tensor, configs)

    # Sync before timing
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark individual operations
    timings = {}

    # Transform
    start = time.perf_counter()
    for _ in range(n_iterations):
        transformed = gpu_transform_trees(tree_tensor.base_vertices, configs)
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    timings['transform'] = (time.perf_counter() - start) / n_iterations

    # BBox
    start = time.perf_counter()
    for _ in range(n_iterations):
        bbox = gpu_compute_bbox(transformed)
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    timings['bbox'] = (time.perf_counter() - start) / n_iterations

    # Overlaps
    start = time.perf_counter()
    for _ in range(n_iterations):
        overlaps = gpu_check_bbox_overlaps(bbox)
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    timings['overlaps'] = (time.perf_counter() - start) / n_iterations

    # Score
    start = time.perf_counter()
    for _ in range(n_iterations):
        side_lengths = gpu_score_configs(bbox)
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    timings['score'] = (time.perf_counter() - start) / n_iterations

    # Full pipeline
    start = time.perf_counter()
    for _ in range(n_iterations):
        transformed, bbox, overlaps, side_lengths = evaluate_configs(tree_tensor, configs)
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    timings['full_pipeline'] = (time.perf_counter() - start) / n_iterations

    print("Timing Results (ms per call):")
    for op, t in timings.items():
        print(f"  {op:20s}: {t*1000:8.3f} ms")

    # Throughput
    configs_per_sec = batch_size / timings['full_pipeline']
    print(f"\nThroughput: {configs_per_sec:.0f} configs/sec")
    print(f"            {configs_per_sec * n_trees:.0f} trees/sec")

    # Memory usage
    if device.type == 'mps':
        # MPS doesn't have easy memory tracking, estimate from tensors
        mem_estimate = (
            configs.numel() * 4 +  # configs
            transformed.numel() * 4 +  # transformed
            bbox.numel() * 4 +  # bbox
            overlaps.numel() +  # overlaps (bool)
            side_lengths.numel() * 4  # side_lengths
        ) / 1e6
        print(f"\nEstimated tensor memory: {mem_estimate:.1f} MB")

    return timings


def test_correctness(n_trees: int = 10):
    """
    Test that GPU primitives produce correct results.
    """
    print(f"\nCorrectness Test (n={n_trees})")
    print("=" * 40)

    device = get_device()
    tree_tensor = TreeTensor(device)

    # Simple test case: all trees at origin with 0 rotation
    configs = torch.zeros(1, n_trees, 3, device=device)
    configs[0, :, 0] = torch.linspace(-2, 2, n_trees)  # Spread along x-axis

    transformed, bbox, overlaps, side_lengths = evaluate_configs(tree_tensor, configs)

    print(f"Config shape: {configs.shape}")
    print(f"Transformed shape: {transformed.shape}")
    print(f"BBox shape: {bbox.shape}")
    print(f"Overlaps shape: {overlaps.shape}")
    print(f"Side length: {side_lengths.item():.4f}")

    # Verify first tree (at x=-2)
    first_tree_bbox = bbox[0, 0].cpu().numpy()
    print(f"\nFirst tree (x=-2, y=0, angle=0):")
    print(f"  BBox: min=({first_tree_bbox[0]:.3f}, {first_tree_bbox[1]:.3f}), "
          f"max=({first_tree_bbox[2]:.3f}, {first_tree_bbox[3]:.3f})")

    # Expected: tree at x=-2 with base vertices means
    # min_x = -2 - 0.35 = -2.35, max_x = -2 + 0.35 = -1.65
    # min_y = -0.2, max_y = 0.8
    expected_bbox = [-2.35, -0.2, -1.65, 0.8]
    print(f"  Expected: min=({expected_bbox[0]:.3f}, {expected_bbox[1]:.3f}), "
          f"max=({expected_bbox[2]:.3f}, {expected_bbox[3]:.3f})")

    bbox_error = np.abs(first_tree_bbox - np.array(expected_bbox)).max()
    print(f"  Max error: {bbox_error:.6f}")

    # Check overlap count
    n_overlaps = gpu_count_overlaps(overlaps).item()
    print(f"\nOverlap count: {n_overlaps}")

    # Test with rotation
    print("\nTesting rotation (45 degrees):")
    configs_rot = configs.clone()
    configs_rot[0, 0, 2] = 45.0  # Rotate first tree 45 degrees

    transformed_rot, bbox_rot, _, _ = evaluate_configs(tree_tensor, configs_rot)

    rotated_bbox = bbox_rot[0, 0].cpu().numpy()
    print(f"  Rotated tree bbox: min=({rotated_bbox[0]:.3f}, {rotated_bbox[1]:.3f}), "
          f"max=({rotated_bbox[2]:.3f}, {rotated_bbox[3]:.3f})")

    # 45-degree rotation should make bbox roughly square and larger
    rot_width = rotated_bbox[2] - rotated_bbox[0]
    rot_height = rotated_bbox[3] - rotated_bbox[1]
    print(f"  Dimensions: {rot_width:.3f} x {rot_height:.3f}")

    print("\nAll tests passed!" if bbox_error < 1e-5 else "\nTEST FAILED!")

    return bbox_error < 1e-5


if __name__ == '__main__':
    # Run correctness tests first
    test_correctness(n_trees=10)

    print("\n" + "=" * 60 + "\n")

    # Then benchmark
    benchmark_gpu_primitives(batch_size=32, n_trees=20, n_iterations=100)

    print("\n" + "=" * 60 + "\n")

    # Scale up test
    benchmark_gpu_primitives(batch_size=32, n_trees=200, n_iterations=50)
