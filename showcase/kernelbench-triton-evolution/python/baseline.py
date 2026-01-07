#!/usr/bin/env python3
"""
Generate baseline timings for PyTorch reference implementations.

Run this on your hardware to get accurate baseline timings.

Usage:
    python baseline.py [--output results/baselines.json]
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def benchmark_pytorch_softmax(
    batch_size: int,
    seq_len: int,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Benchmark PyTorch softmax."""
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        _ = F.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = F.softmax(x, dim=-1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = sorted(times)

    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "max_ms": times[-1],
        "iterations": iterations,
    }


def generate_baselines(output_path: Path):
    """Generate baseline timings for all tasks."""
    print("GPU Info:")
    gpu_info = get_gpu_info()
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")
    print()

    baselines = {
        "_meta": {
            "gpu": gpu_info,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "softmax": {},
    }

    # Softmax benchmark shapes
    shapes = [
        (1, 128),
        (32, 512),
        (64, 1024),
        (64, 2048),
        (128, 2048),
        (128, 4096),
        (256, 4096),
    ]

    print("Benchmarking PyTorch softmax...")
    for batch_size, seq_len in shapes:
        shape_key = f"{batch_size}x{seq_len}"
        print(f"  {shape_key}...", end=" ", flush=True)

        result = benchmark_pytorch_softmax(batch_size, seq_len)
        baselines["softmax"][shape_key] = result

        print(f"{result['median_ms']:.3f}ms")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\nBaselines saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline timings")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "baselines.json",
        help="Output path for baselines",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Cannot generate baselines.")
        return 1

    generate_baselines(args.output)
    return 0


if __name__ == "__main__":
    exit(main())
