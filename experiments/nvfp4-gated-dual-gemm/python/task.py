"""
NVFP4 Gated Dual GEMM Task Definition

Problem: C = silu(A @ B1) * (A @ B2)

Input: NVFP4 quantized matrices with FP8 scale factors
Output: Float16 result
"""
import torch
from typing import Tuple, TypeVar, TypedDict

# Type definitions
T = TypeVar("T", bound=torch.Tensor)

# Input: tuple of 10 tensors
# (a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_permuted, sfb1_permuted, sfb2_permuted, c)
input_t = Tuple[T, T, T, T, T, T, T, T, T, T]

# Output: single tensor (modified c)
output_t = T


class TestSpec(TypedDict):
    """Test case specification"""
    m: int      # Number of rows in A and C
    n: int      # Number of columns in B and C
    k: int      # Inner dimension (columns of A, rows of B)
    l: int      # Batch dimension
    seed: int   # Random seed for reproducibility


# Scale factor configuration
SF_VEC_SIZE = 16  # One FP8 scale per 16 FP4 values


def ceil_div(a: int, b: int) -> int:
    """Ceiling division"""
    return (a + b - 1) // b


# Benchmark test cases (representative of real workloads)
TRAIN_SPECS = [
    TestSpec(m=1024, n=1024, k=2048, l=1, seed=42),
    TestSpec(m=2048, n=2048, k=4096, l=1, seed=43),
    TestSpec(m=4096, n=4096, k=4096, l=1, seed=44),
]

VALID_SPECS = [
    TestSpec(m=1536, n=1536, k=3072, l=1, seed=100),
    TestSpec(m=3072, n=3072, k=4096, l=1, seed=101),
]

TEST_SPECS = [
    TestSpec(m=2048, n=4096, k=4096, l=1, seed=200),
    TestSpec(m=4096, n=2048, k=4096, l=1, seed=201),
    TestSpec(m=8192, n=8192, k=4096, l=1, seed=202),
]


def generate_nvfp4_tensor(shape: Tuple[int, ...], seed: int, device: str = "cuda"):
    """
    Generate NVFP4 quantized tensor with scale factors.

    Returns:
        data: FP4 quantized values
        scale_cpu: FP8 scale factors (CPU reference)
        scale_permuted: FP8 scale factors (permuted for kernel)
    """
    torch.manual_seed(seed)

    # Generate random FP32 source data
    fp32_data = torch.randn(shape, dtype=torch.float32, device=device)

    # Quantize to FP4 (simulated - actual FP4 uses torch.float4_e2m1)
    # For now, use FP16 as placeholder until running on actual Blackwell
    fp4_data = fp32_data.to(torch.float16)

    # Generate scale factors (one per 16 elements along last dimension)
    sf_shape = list(shape)
    sf_shape[-1] = ceil_div(shape[-1], SF_VEC_SIZE)

    # Scale factors in FP8 (simulated with FP16 for compatibility)
    scale_cpu = torch.ones(sf_shape, dtype=torch.float16, device="cpu")

    # Permuted scale factors for efficient kernel access
    # Layout: [32, 4, rest_dim0, 4, rest_dim1, ...]
    scale_permuted = scale_cpu.clone()  # Placeholder - actual permutation TBD

    return fp4_data, scale_cpu, scale_permuted


def generate_test_data(spec: TestSpec, device: str = "cuda") -> input_t:
    """Generate input data for a test case"""
    m, n, k, l = spec["m"], spec["n"], spec["k"], spec["l"]
    seed = spec["seed"]

    # Generate A matrix
    a, sfa_cpu, sfa_permuted = generate_nvfp4_tensor((m, k, l), seed)

    # Generate B1 matrix
    b1, sfb1_cpu, sfb1_permuted = generate_nvfp4_tensor((n, k, l), seed + 1000)

    # Generate B2 matrix
    b2, sfb2_cpu, sfb2_permuted = generate_nvfp4_tensor((n, k, l), seed + 2000)

    # Allocate output tensor
    c = torch.empty((m, n, l), dtype=torch.float16, device=device)

    return (a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu,
            sfa_permuted, sfb1_permuted, sfb2_permuted, c)


def check_correctness(output: output_t, reference: output_t, rtol: float = 1e-2, atol: float = 1e-3) -> Tuple[bool, float]:
    """
    Check if output matches reference within tolerance.

    Returns:
        (is_correct, max_error)
    """
    diff = (output - reference).abs()
    max_error = diff.max().item()

    # Relative and absolute tolerance check
    is_correct = torch.allclose(output, reference, rtol=rtol, atol=atol)

    return is_correct, max_error
