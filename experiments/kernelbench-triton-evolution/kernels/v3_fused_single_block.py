"""
V3: Fused Single-Block Softmax

For sequences that fit in one block (n_cols <= BLOCK_SIZE),
use a true single-pass algorithm with all computation in registers.

This is the theoretical best case: 1 read, 1 write, no rescaling.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def softmax_kernel_v3_small(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    True single-pass softmax for small sequences.

    BLOCK_SIZE must be >= n_cols for this kernel to work correctly.
    All computation happens in registers - no global memory re-reads.
    """
    LOG2E = 1.4426950408889634

    row_idx = tl.program_id(0)

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Single read into registers
    vals = tl.load(input_row_start + col_offsets, mask=mask, other=float('-inf'))

    # All computation in registers
    max_val = tl.max(vals, axis=0)
    exp_vals = tl.exp2((vals - max_val) * LOG2E)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp

    # Single write
    tl.store(output_row_start + col_offsets, softmax_vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel_v3_large(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Tiled softmax for larger sequences."""
    LOG2E = 1.4426950408889634

    row_idx = tl.program_id(0)

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    running_max = float('-inf')
    running_sum = 0.0

    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        vals = tl.load(input_row_start + col_offsets, mask=mask, other=float('-inf'))

        block_max = tl.max(vals, axis=0)
        new_max = tl.maximum(running_max, block_max)

        scale = tl.exp2((running_max - new_max) * LOG2E)
        running_sum = running_sum * scale

        exp_vals = tl.exp2((vals - new_max) * LOG2E)
        running_sum = running_sum + tl.sum(exp_vals, axis=0)

        running_max = new_max

    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        vals = tl.load(input_row_start + col_offsets, mask=mask, other=0.0)
        exp_vals = tl.exp2((vals - running_max) * LOG2E)
        softmax_vals = exp_vals / running_sum

        tl.store(output_row_start + col_offsets, softmax_vals, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized softmax with dispatch based on sequence length.

    For small sequences (n_cols <= 4096), use true single-pass.
    For larger sequences, use tiled online algorithm.
    """
    assert x.dim() == 2, "Input must be 2D (batch, seq_len)"
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)
    grid = (n_rows,)

    # Dispatch based on size
    if n_cols <= 256:
        softmax_kernel_v3_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=256
        )
    elif n_cols <= 512:
        softmax_kernel_v3_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=512
        )
    elif n_cols <= 1024:
        softmax_kernel_v3_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=1024
        )
    elif n_cols <= 2048:
        softmax_kernel_v3_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=2048
        )
    elif n_cols <= 4096:
        softmax_kernel_v3_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=4096
        )
    else:
        softmax_kernel_v3_large[grid](
            x, y, x.stride(0), y.stride(0), n_cols
        )

    return y


# Aliases
optimized_softmax = softmax
softmax_triton = softmax
triton_softmax = softmax


if __name__ == "__main__":
    import torch.nn.functional as F
    import time

    print("V3 Fused Single-Block Softmax")
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
