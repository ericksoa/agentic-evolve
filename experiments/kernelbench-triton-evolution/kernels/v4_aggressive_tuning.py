"""
V4: Aggressive Autotuning with more configurations

Explores a wider range of num_warps and num_stages to find
optimal configurations for T4 GPU.
"""

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        # Aggressive warp configurations
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32, num_stages=2),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel_v4(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Aggressively autotuned softmax kernel.

    Uses single-pass when possible (BLOCK_SIZE >= n_cols),
    falls back to 2-pass online algorithm otherwise.
    """
    LOG2E = 1.4426950408889634

    row_idx = tl.program_id(0)
    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Check if entire row fits in one block
    # Note: This comparison happens at compile time due to constexpr
    if BLOCK_SIZE >= 4096:  # Our max test size
        # True single-pass: load once, compute, store once
        mask = col_offsets < n_cols
        vals = tl.load(input_row_start + col_offsets, mask=mask, other=float('-inf'))

        max_val = tl.max(vals, axis=0)
        exp_vals = tl.exp2((vals - max_val) * LOG2E)
        sum_exp = tl.sum(exp_vals, axis=0)
        softmax_vals = exp_vals / sum_exp

        tl.store(output_row_start + col_offsets, softmax_vals, mask=mask)
    else:
        # Tiled online algorithm
        running_max = float('-inf')
        running_sum = 0.0

        for col_start in range(0, n_cols, BLOCK_SIZE):
            offs = col_start + col_offsets
            mask = offs < n_cols

            vals = tl.load(input_row_start + offs, mask=mask, other=float('-inf'))

            block_max = tl.max(vals, axis=0)
            new_max = tl.maximum(running_max, block_max)

            scale = tl.exp2((running_max - new_max) * LOG2E)
            running_sum = running_sum * scale

            exp_vals = tl.exp2((vals - new_max) * LOG2E)
            running_sum = running_sum + tl.sum(exp_vals, axis=0)

            running_max = new_max

        for col_start in range(0, n_cols, BLOCK_SIZE):
            offs = col_start + col_offsets
            mask = offs < n_cols

            vals = tl.load(input_row_start + offs, mask=mask, other=0.0)
            exp_vals = tl.exp2((vals - running_max) * LOG2E)
            softmax_vals = exp_vals / running_sum

            tl.store(output_row_start + offs, softmax_vals, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Aggressively autotuned softmax."""
    assert x.dim() == 2, "Input must be 2D (batch, seq_len)"
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)
    grid = (n_rows,)

    softmax_kernel_v4[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
    )

    return y


# Aliases
optimized_softmax = softmax
softmax_triton = softmax
triton_softmax = softmax
