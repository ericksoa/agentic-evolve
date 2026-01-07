"""
V5: Explicit Dispatch with Specialized Kernels

Instead of relying on autotune, use explicit dispatch based on input size.
Each kernel is specialized for a specific size range.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def softmax_small(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass for small sequences."""
    LOG2E = 1.4426950408889634
    row_idx = tl.program_id(0)

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    vals = tl.load(input_row_start + col_offsets, mask=mask, other=float('-inf'))
    max_val = tl.max(vals, axis=0)
    exp_vals = tl.exp2((vals - max_val) * LOG2E)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    tl.store(output_row_start + col_offsets, softmax_vals, mask=mask)


@triton.jit
def softmax_medium(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Two-pass online for medium sequences."""
    LOG2E = 1.4426950408889634
    row_idx = tl.program_id(0)

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    running_max = float('-inf')
    running_sum = 0.0

    # First pass
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

    # Second pass
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offs = col_start + col_offsets
        mask = offs < n_cols
        vals = tl.load(input_row_start + offs, mask=mask, other=0.0)
        exp_vals = tl.exp2((vals - running_max) * LOG2E)
        softmax_vals = exp_vals / running_sum
        tl.store(output_row_start + offs, softmax_vals, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Explicit dispatch based on sequence length."""
    assert x.dim() == 2, "Input must be 2D (batch, seq_len)"
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)
    grid = (n_rows,)

    # Explicit dispatch - no autotune overhead
    if n_cols <= 1024:
        softmax_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols,
            BLOCK_SIZE=1024, num_warps=8
        )
    elif n_cols <= 2048:
        softmax_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols,
            BLOCK_SIZE=2048, num_warps=8
        )
    elif n_cols <= 4096:
        softmax_small[grid](
            x, y, x.stride(0), y.stride(0), n_cols,
            BLOCK_SIZE=4096, num_warps=16
        )
    else:
        softmax_medium[grid](
            x, y, x.stride(0), y.stride(0), n_cols,
            BLOCK_SIZE=1024, num_warps=8
        )

    return y


# Aliases
optimized_softmax = softmax
softmax_triton = softmax
triton_softmax = softmax
