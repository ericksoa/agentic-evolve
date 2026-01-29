"""
NVFP4 Gated Dual GEMM - Evolution Starter Kernel

Computes: C = silu(A @ B1) * (A @ B2)

This is the starting point for evolution. The goal is to optimize
this kernel to approach speed-of-light performance on Blackwell GPUs.

Optimization targets:
- Tile sizes (M, N, K)
- Memory hierarchy (TMEM, shared memory)
- Warp specialization
- Pipeline depth
- Fused operations (silu + multiply)
"""
import torch
import torch.nn.functional as F
from task import input_t, output_t

# Kernel configuration (mutation targets)
TILE_M = 128
TILE_N = 128
TILE_K = 256
MMA_K = 64
SF_BLOCK_SIZE = 16
THREADS_PER_BLOCK = 128
SHARED_MEM_STAGES = 1


# Compiled kernel cache
_compiled_kernel = None


def compile_kernel():
    """
    Compile the kernel once and cache for reuse.
    This avoids recompilation overhead during benchmarking.
    """
    global _compiled_kernel
    if _compiled_kernel is not None:
        return _compiled_kernel

    # For now, return the simple implementation
    # This will be replaced with Triton/CUDA kernel during evolution
    _compiled_kernel = fused_dual_gemm_silu
    return _compiled_kernel


def fused_dual_gemm_silu(
    a: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b1: torch.Tensor,
    scale_b2: torch.Tensor,
) -> torch.Tensor:
    """
    Fused Dual GEMM with SiLU activation.

    Gen 0: Simple PyTorch implementation (baseline to beat)

    Computes: C = silu(A @ B1) * (A @ B2)
    """
    # Transpose B matrices for matmul
    b1_t = b1.transpose(-2, -1)
    b2_t = b2.transpose(-2, -1)

    # First GEMM: A @ B1
    result1 = torch.matmul(a.float(), b1_t.float())

    # Second GEMM: A @ B2
    result2 = torch.matmul(a.float(), b2_t.float())

    # Fused activation: silu(result1) * result2
    output = (F.silu(result1) * result2).to(torch.float16)

    return output


def custom_kernel(data: input_t) -> output_t:
    """
    Main entry point for the NVFP4 Gated Dual GEMM kernel.

    Input: tuple of 10 tensors
        - a: [m, k, l] - Input matrix A in NVFP4
        - b1: [n, k, l] - Weight matrix B1 in NVFP4
        - b2: [n, k, l] - Weight matrix B2 in NVFP4
        - sfa_cpu: [m, k//16, l] - Scale factors for A (CPU reference)
        - sfb1_cpu: [n, k//16, l] - Scale factors for B1 (CPU reference)
        - sfb2_cpu: [n, k//16, l] - Scale factors for B2 (CPU reference)
        - sfa_permuted: Permuted scale factors for A (kernel optimized)
        - sfb1_permuted: Permuted scale factors for B1 (kernel optimized)
        - sfb2_permuted: Permuted scale factors for B2 (kernel optimized)
        - c: [m, n, l] - Output buffer

    Output: [m, n, l] Float16 tensor
    """
    (a, b1, b2,
     sfa_cpu, sfb1_cpu, sfb2_cpu,
     sfa_permuted, sfb1_permuted, sfb2_permuted,
     c) = data

    m, n, l = c.shape
    kernel_fn = compile_kernel()

    # Process batches
    for l_idx in range(l):
        result = kernel_fn(
            a[:, :, l_idx],
            b1[:, :, l_idx],
            b2[:, :, l_idx],
            sfa_permuted,  # Use permuted for kernel
            sfb1_permuted,
            sfb2_permuted,
        )
        c[:, :, l_idx] = result

    return c


# =============================================================================
# EVOLUTION TARGETS BELOW
# The following functions represent different optimization strategies
# that the evolution process will explore and combine.
# =============================================================================

def triton_dual_gemm_v1(a, b1, b2, sfa, sfb1, sfb2):
    """
    Triton implementation placeholder.

    Optimization strategy: Tiled computation with shared memory staging.
    """
    # TODO: Implement Triton kernel
    # import triton
    # import triton.language as tl
    #
    # @triton.jit
    # def dual_gemm_silu_kernel(...)
    #     ...
    pass


def cutlass_dual_gemm_v1(a, b1, b2, sfa, sfb1, sfb2):
    """
    CuTe DSL / CUTLASS implementation placeholder.

    Optimization strategy:
    - Warp specialization (separate MMA and epilogue warps)
    - TMEM for Blackwell
    - Block-scaled tensor cores
    """
    # TODO: Implement CUTLASS kernel via CuTe DSL
    pass


# =============================================================================
# OPTIMIZATION NOTES FOR EVOLUTION
# =============================================================================
#
# Key performance factors for NVFP4 Gated Dual GEMM:
#
# 1. MEMORY BANDWIDTH
#    - FP4 format reduces memory traffic by 4x vs FP16
#    - Scale factor overhead: 1 FP8 per 16 values = 6.25% overhead
#    - Shared memory staging hides global memory latency
#
# 2. COMPUTE UTILIZATION
#    - Blackwell tensor cores support native FP4 operations
#    - Two GEMMs can share A matrix (data reuse)
#    - Fusing silu reduces memory round-trips
#
# 3. TILE SIZE SELECTION
#    - (128, 128, 256) is baseline - optimized for SM100
#    - Larger tiles: Better compute density, worse occupancy
#    - Smaller tiles: Better occupancy, more overhead
#
# 4. PIPELINE DEPTH
#    - Double/triple buffering hides memory latency
#    - More stages = more shared memory = lower occupancy
#
# 5. WARP SPECIALIZATION
#    - Separate producer (memory) and consumer (compute) warps
#    - Reduces register pressure and improves throughput
#
# =============================================================================
