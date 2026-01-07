"""
Starter Triton softmax kernel.

Based on Triton's official fused softmax tutorial.
This serves as the initial seed for evolution.

Reference: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel.

    Each program handles one row of the input.
    """
    # Get row index
    row_idx = tl.program_id(0)

    # Compute pointers to the row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load row with masking for out-of-bounds
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # Compute softmax (numerically stable)
    # Step 1: Subtract max for numerical stability
    row_max = tl.max(row, axis=0)
    row_stable = row - row_max

    # Step 2: Compute exp
    numerator = tl.exp(row_stable)

    # Step 3: Compute sum
    denominator = tl.sum(numerator, axis=0)

    # Step 4: Normalize
    softmax_output = numerator / denominator

    # Store output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax using Triton kernel.

    Args:
        x: Input tensor of shape (batch_size, seq_len)

    Returns:
        Softmax output of same shape
    """
    n_rows, n_cols = x.shape

    # Determine block size (must be power of 2, >= n_cols)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Allocate output
    y = torch.empty_like(x)

    # Launch kernel - one program per row
    softmax_kernel[(n_rows,)](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


# Alias for benchmark harness
kernel = softmax_triton


if __name__ == "__main__":
    import torch.nn.functional as F

    # Test correctness
    torch.manual_seed(42)
    x = torch.randn(32, 1024, device="cuda", dtype=torch.float32)

    triton_out = softmax_triton(x)
    torch_out = F.softmax(x, dim=-1)

    max_diff = (triton_out - torch_out).abs().max().item()
    print(f"Max difference vs PyTorch: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check FAILED!")

    # Quick benchmark
    import time

    # Warmup
    for _ in range(10):
        _ = softmax_triton(x)
    torch.cuda.synchronize()

    # Time it
    start = time.perf_counter()
    for _ in range(100):
        _ = softmax_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 100 * 1000

    # Time PyTorch
    for _ in range(10):
        _ = F.softmax(x, dim=-1)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = F.softmax(x, dim=-1)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / 100 * 1000

    print(f"\nTiming (32x1024):")
    print(f"  Triton: {triton_time:.3f}ms")
    print(f"  PyTorch: {torch_time:.3f}ms")
    print(f"  Speedup: {torch_time/triton_time:.2f}x")
