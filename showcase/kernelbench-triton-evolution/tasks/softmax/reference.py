"""
PyTorch reference implementation for softmax.

This is the baseline we're trying to beat with Triton.
"""

import torch
import torch.nn.functional as F


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """
    Reference softmax implementation using PyTorch.

    Args:
        x: Input tensor of shape (batch_size, seq_len)

    Returns:
        Softmax output of same shape, normalized along dim=-1
    """
    return F.softmax(x, dim=-1)


def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """
    Naive softmax implementation (for understanding).

    This shows the algorithm explicitly:
    1. Subtract max for numerical stability
    2. Compute exp
    3. Normalize by sum
    """
    # Subtract max for numerical stability
    x_max = x.max(dim=-1, keepdim=True).values
    x_stable = x - x_max

    # Compute exp
    exp_x = torch.exp(x_stable)

    # Normalize
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


# Test equivalence
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(32, 1024, device="cuda")

    ref = softmax_reference(x)
    naive = softmax_naive(x)

    print(f"Max difference: {(ref - naive).abs().max().item():.2e}")
    assert torch.allclose(ref, naive, atol=1e-5), "Implementations don't match!"
    print("✓ Reference implementations match")
