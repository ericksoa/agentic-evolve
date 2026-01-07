#!/usr/bin/env python3
"""
Quick test script for MoE kernels.
Run on T4 to validate correctness before H200 benchmarking.

Usage:
    python test_moe.py          # Correctness tests
    python test_moe.py --bench  # Correctness + benchmark
"""

import torch
import sys
from pathlib import Path

# Add kernels to path
sys.path.insert(0, str(Path(__file__).parent / "kernels"))

from baseline_moe import moe_forward_baseline
from colmajor_moe import moe_forward_colmajor
from splitk_moe import moe_forward_splitk
from padding_free_moe import moe_forward_padding_free
from optimized_moe import moe_forward_optimized


def test_correctness():
    """Test that all kernels produce same output as baseline."""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    device = torch.device('cuda')
    dtype = torch.float16

    # Test configurations (small for quick verification)
    configs = [
        {"batch": 1, "seq": 8, "hidden": 256, "intermediate": 128, "experts": 16, "top_k": 4},
        {"batch": 2, "seq": 16, "hidden": 512, "intermediate": 256, "experts": 32, "top_k": 4},
        {"batch": 4, "seq": 32, "hidden": 256, "intermediate": 128, "experts": 64, "top_k": 8},
    ]

    kernels = {
        "colmajor": moe_forward_colmajor,
        "splitk": moe_forward_splitk,
        "padding_free": moe_forward_padding_free,
        "optimized": moe_forward_optimized,
    }

    all_passed = True

    for cfg in configs:
        print(f"\nConfig: B={cfg['batch']}, S={cfg['seq']}, E={cfg['experts']}, K={cfg['top_k']}")

        # Create inputs
        hidden = torch.randn(cfg['batch'], cfg['seq'], cfg['hidden'], device=device, dtype=dtype)
        router = torch.randn(cfg['batch'], cfg['seq'], cfg['experts'], device=device, dtype=dtype)
        gate_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
        up_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
        down_w = torch.randn(cfg['experts'], cfg['intermediate'], cfg['hidden'], device=device, dtype=dtype)

        # Baseline reference
        baseline_out = moe_forward_baseline(hidden, router, gate_w, up_w, down_w, top_k=cfg['top_k'])

        # Test each kernel
        for name, kernel in kernels.items():
            try:
                out = kernel(hidden, router, gate_w, up_w, down_w, top_k=cfg['top_k'])
                max_diff = (out - baseline_out).abs().max().item()
                rel_diff = max_diff / (baseline_out.abs().mean().item() + 1e-6)

                if max_diff < 0.1:  # Allow some numerical tolerance
                    print(f"  {name:<15} PASS  (max_diff={max_diff:.6f}, rel={rel_diff:.4f})")
                else:
                    print(f"  {name:<15} FAIL  (max_diff={max_diff:.6f}, rel={rel_diff:.4f})")
                    all_passed = False
            except Exception as e:
                print(f"  {name:<15} ERROR: {e}")
                all_passed = False

    return all_passed


def test_benchmark(mode="quick"):
    """Quick benchmark to verify performance trends."""
    import time

    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    device = torch.device('cuda')
    dtype = torch.float16

    # Benchmark configs
    if mode == "quick":
        # Small configs for T4
        configs = [
            {"batch": 1, "seq": 1, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
            {"batch": 8, "seq": 1, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
            {"batch": 1, "seq": 128, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
            {"batch": 8, "seq": 128, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
        ]
        num_warmup, num_iter = 5, 20
    else:
        # Larger configs for H200
        configs = [
            {"batch": 1, "seq": 1, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8},
            {"batch": 8, "seq": 1, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8},
            {"batch": 1, "seq": 512, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8},
            {"batch": 8, "seq": 512, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8},
        ]
        num_warmup, num_iter = 10, 50

    kernels = {
        "baseline": moe_forward_baseline,
        "colmajor": moe_forward_colmajor,
        "padding_free": moe_forward_padding_free,
        "optimized": moe_forward_optimized,
    }

    print(f"\n{'Config':<30} " + " ".join(f"{k:<12}" for k in kernels.keys()) + " Speedup")
    print("-" * 90)

    for cfg in configs:
        try:
            hidden = torch.randn(cfg['batch'], cfg['seq'], cfg['hidden'], device=device, dtype=dtype)
            router = torch.randn(cfg['batch'], cfg['seq'], cfg['experts'], device=device, dtype=dtype)
            gate_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
            up_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
            down_w = torch.randn(cfg['experts'], cfg['intermediate'], cfg['hidden'], device=device, dtype=dtype)

            results = {}
            for name, kernel in kernels.items():
                try:
                    # Warmup
                    for _ in range(num_warmup):
                        _ = kernel(hidden, router, gate_w, up_w, down_w, top_k=cfg['top_k'])
                    torch.cuda.synchronize()

                    # Benchmark
                    start = time.perf_counter()
                    for _ in range(num_iter):
                        _ = kernel(hidden, router, gate_w, up_w, down_w, top_k=cfg['top_k'])
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) / num_iter * 1000
                    results[name] = elapsed
                except Exception as e:
                    results[name] = None

            # Print results
            config_str = f"B{cfg['batch']}S{cfg['seq']}E{cfg['experts']}K{cfg['top_k']}"
            times_str = " ".join(
                f"{results[k]:<12.2f}" if results[k] else f"{'ERROR':<12}"
                for k in kernels.keys()
            )

            # Calculate speedup vs baseline
            if results.get("baseline") and results.get("optimized"):
                speedup = results["baseline"] / results["optimized"]
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{config_str:<30} {times_str} {speedup_str}")

        except Exception as e:
            print(f"Config {cfg} failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test MoE kernels")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--full", action="store_true", help="Full benchmark (H200 configs)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Always run correctness tests
    passed = test_correctness()

    if args.bench:
        mode = "full" if args.full else "quick"
        test_benchmark(mode=mode)

    print("\n" + "=" * 60)
    if passed:
        print("All correctness tests PASSED")
    else:
        print("Some correctness tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
