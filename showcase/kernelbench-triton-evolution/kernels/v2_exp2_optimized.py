"""
V2: Optimized Softmax with exp2 and better memory patterns

Optimizations over V1:
1. Use exp2 instead of exp (faster on NVIDIA GPUs)
   exp(x) = exp2(x * log2(e)) = exp2(x * 1.4426950408889634)
2. Better autotune configurations
3. Explicit num_stages for pipelining
"""

import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        # Smaller blocks with more warps for better occupancy
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel_v2(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized online softmax with exp2.

    exp(x - max) = exp2((x - max) * log2(e))

    exp2 is faster than exp on NVIDIA GPUs because it maps
    directly to the hardware instruction.
    """
    # log2(e) = 1.4426950408889634 (inlined to avoid global variable issues)
    LOG2E = 1.4426950408889634

    row_idx = tl.program_id(0)

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    # Running statistics
    running_max = float('-inf')
    running_sum = 0.0

    # First pass: compute max and sum using online algorithm
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        vals = tl.load(input_row_start + col_offsets, mask=mask, other=float('-inf'))

        block_max = tl.max(vals, axis=0)
        new_max = tl.maximum(running_max, block_max)

        # Rescale using exp2: exp(old - new) = exp2((old - new) * log2(e))
        scale = tl.exp2((running_max - new_max) * LOG2E)
        running_sum = running_sum * scale

        # exp(vals - new_max) = exp2((vals - new_max) * log2(e))
        exp_vals = tl.exp2((vals - new_max) * LOG2E)
        running_sum = running_sum + tl.sum(exp_vals, axis=0)

        running_max = new_max

    # Second pass: compute output
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        vals = tl.load(input_row_start + col_offsets, mask=mask, other=0.0)
        exp_vals = tl.exp2((vals - running_max) * LOG2E)
        softmax_vals = exp_vals / running_sum

        tl.store(output_row_start + col_offsets, softmax_vals, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Optimized softmax with exp2."""
    assert x.dim() == 2, "Input must be 2D (batch, seq_len)"
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)
    grid = (n_rows,)

    softmax_kernel_v2[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
    )

    return y


# Aliases for evaluator
optimized_softmax = softmax
softmax_triton = softmax
triton_softmax = softmax


if __name__ == "__main__":
    import torch.nn.functional as F
    import time

    print("V2 exp2-Optimized Softmax")
    print("=" * 50)

    # Test correctness
    print("\nCorrectness Test:")
    for shape in [(32, 128), (256, 4096), (1024, 4096), (512, 8192)]:
        x = torch.randn(shape, device='cuda', dtype=torch.float32)
        y_ours = softmax(x)
        y_ref = F.softmax(x, dim=-1)
        max_diff = torch.max(torch.abs(y_ours - y_ref)).item()
        status = "PASS" if max_diff < 1e-5 else "FAIL"
        print(f"  {shape}: max_diff={max_diff:.2e} [{status}]")

    # Benchmark
    print("\nBenchmark:")
    for shape in [(256, 4096), (512, 4096), (1024, 2048), (1024, 4096)]:
        x = torch.randn(shape, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(20):
            _ = softmax(x)
            _ = F.softmax(x, dim=-1)
        torch.cuda.synchronize()

        # Time ours
        start = time.perf_counter()
        for _ in range(100):
            _ = softmax(x)
        torch.cuda.synchronize()
        our_time = (time.perf_counter() - start) / 100 * 1000

        # Time PyTorch
        start = time.perf_counter()
        for _ in range(100):
            _ = F.softmax(x, dim=-1)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000

        speedup = pytorch_time / our_time
        print(f"  {shape}: ours={our_time:.4f}ms, pytorch={pytorch_time:.4f}ms, speedup={speedup:.2f}x")
