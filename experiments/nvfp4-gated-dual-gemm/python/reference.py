"""
Reference Implementation for NVFP4 Gated Dual GEMM

Computes: C = silu(A @ B1) * (A @ B2)

This is a slow but correct reference using PyTorch's scaled_mm.
"""
import torch
import torch.nn.functional as F
from task import input_t, output_t, ceil_div


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert scale factor tensor to blocked format for _scaled_mm.

    The blocked format aligns with CUTLASS's expected memory layout
    for block-scaled operations.
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Pad if necessary
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if rows < padded_rows or cols < padded_cols:
        padded = torch.zeros((padded_rows, padded_cols),
                             dtype=input_matrix.dtype,
                             device=input_matrix.device)
        padded[:rows, :cols] = input_matrix
    else:
        padded = input_matrix

    # Reshape into blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)

    # Rearrange for blocked access
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of NVFP4 Gated Dual GEMM.

    Computes: C = silu(A @ B1) * (A @ B2)

    This uses PyTorch's _scaled_mm for correctness verification.
    It's not optimized and processes batches sequentially.
    """
    (a_ref, b1_ref, b2_ref,
     sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu,
     _, _, _, c_ref) = data

    m, n, l = c_ref.shape

    # Allocate intermediate results in FP32 for numerical stability
    ref1 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)
    ref2 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)

    # Process each batch
    for l_idx in range(l):
        # Convert scale factors to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b1 = to_blocked(sfb1_ref_cpu[:, :, l_idx])
        scale_b2 = to_blocked(sfb2_ref_cpu[:, :, l_idx])

        # First GEMM: A @ B1
        res1 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b1_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b1.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref1[:, :, l_idx] = res1

        # Second GEMM: A @ B2
        res2 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b2_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b2.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref2[:, :, l_idx] = res2

    # Apply silu to first result and multiply by second
    # silu(x) = x * sigmoid(x)
    c_result = (F.silu(ref1) * ref2).to(torch.float16)

    return c_result


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid Linear Unit activation.
    silu(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)
