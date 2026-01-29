"""
Test cases for softmax kernel correctness.

A kernel must pass ALL tests to be considered correct (fitness > 0).
"""

import torch
import torch.nn.functional as F
from typing import Callable

# Tolerance for floating point comparison
ATOL = 1e-5
RTOL = 1e-5


def get_test_cases() -> list[dict]:
    """
    Return list of test cases with inputs and expected behavior.
    """
    return [
        # Basic shapes
        {"batch_size": 1, "seq_len": 128, "name": "small"},
        {"batch_size": 32, "seq_len": 512, "name": "medium"},
        {"batch_size": 64, "seq_len": 1024, "name": "large"},
        {"batch_size": 128, "seq_len": 2048, "name": "xlarge"},

        # Edge cases
        {"batch_size": 1, "seq_len": 1, "name": "single_element"},
        {"batch_size": 1, "seq_len": 7, "name": "non_power_of_2"},
        {"batch_size": 16, "seq_len": 127, "name": "odd_seq_len"},
        {"batch_size": 1, "seq_len": 8192, "name": "very_long"},

        # Numerical edge cases (tested separately)
    ]


def generate_input(batch_size: int, seq_len: int, seed: int = 42) -> torch.Tensor:
    """Generate random input tensor."""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)


def generate_extreme_input(batch_size: int, seq_len: int, case: str) -> torch.Tensor:
    """Generate inputs that test numerical stability."""
    if case == "large_values":
        # Large positive values - tests overflow prevention
        return torch.full((batch_size, seq_len), 100.0, device="cuda", dtype=torch.float32)
    elif case == "small_values":
        # Large negative values - tests underflow handling
        return torch.full((batch_size, seq_len), -100.0, device="cuda", dtype=torch.float32)
    elif case == "mixed_extreme":
        # Mix of large and small - tests max subtraction
        x = torch.zeros(batch_size, seq_len, device="cuda", dtype=torch.float32)
        x[:, 0] = 100.0
        x[:, 1:] = -100.0
        return x
    elif case == "zeros":
        return torch.zeros(batch_size, seq_len, device="cuda", dtype=torch.float32)
    else:
        raise ValueError(f"Unknown case: {case}")


def test_correctness(kernel_fn: Callable, verbose: bool = False) -> tuple[bool, str]:
    """
    Test kernel correctness against PyTorch reference.

    Args:
        kernel_fn: Function that takes (x) and returns softmax(x)
        verbose: Print detailed results

    Returns:
        (passed, message) tuple
    """
    all_passed = True
    messages = []

    # Standard test cases
    for case in get_test_cases():
        x = generate_input(case["batch_size"], case["seq_len"])

        try:
            result = kernel_fn(x)
            expected = F.softmax(x, dim=-1)

            if not torch.allclose(result, expected, atol=ATOL, rtol=RTOL):
                max_diff = (result - expected).abs().max().item()
                all_passed = False
                messages.append(f"FAIL {case['name']}: max_diff={max_diff:.2e}")
            elif verbose:
                messages.append(f"PASS {case['name']}")

        except Exception as e:
            all_passed = False
            messages.append(f"ERROR {case['name']}: {str(e)[:100]}")

    # Numerical stability tests
    for extreme_case in ["large_values", "small_values", "mixed_extreme", "zeros"]:
        x = generate_extreme_input(32, 256, extreme_case)

        try:
            result = kernel_fn(x)
            expected = F.softmax(x, dim=-1)

            # Check for NaN/Inf
            if torch.isnan(result).any() or torch.isinf(result).any():
                all_passed = False
                messages.append(f"FAIL {extreme_case}: produced NaN/Inf")
                continue

            if not torch.allclose(result, expected, atol=ATOL, rtol=RTOL):
                max_diff = (result - expected).abs().max().item()
                all_passed = False
                messages.append(f"FAIL {extreme_case}: max_diff={max_diff:.2e}")
            elif verbose:
                messages.append(f"PASS {extreme_case}")

        except Exception as e:
            all_passed = False
            messages.append(f"ERROR {extreme_case}: {str(e)[:100]}")

    # Check softmax properties
    x = generate_input(32, 512)
    try:
        result = kernel_fn(x)

        # Sum should be 1
        row_sums = result.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            all_passed = False
            messages.append(f"FAIL property: rows don't sum to 1")

        # All values should be in [0, 1]
        if result.min() < 0 or result.max() > 1:
            all_passed = False
            messages.append(f"FAIL property: values outside [0,1]")

    except Exception as e:
        all_passed = False
        messages.append(f"ERROR properties: {str(e)[:100]}")

    summary = "PASSED" if all_passed else "FAILED"
    return all_passed, f"{summary}: " + "; ".join(messages) if messages else summary


if __name__ == "__main__":
    # Test the reference implementation
    from reference import softmax_reference

    passed, msg = test_correctness(softmax_reference, verbose=True)
    print(msg)
