"""
Unified MoE Benchmark Suite

Benchmarks different MoE kernel implementations across various configurations.
Designed for both local testing (T4) and production benchmarking (H200).
"""

import torch
import time
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add kernels to path
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

from baseline_moe import moe_forward_baseline
from colmajor_moe import moe_forward_colmajor
from splitk_moe import moe_forward_splitk
from padding_free_moe import moe_forward_padding_free
from optimized_moe import moe_forward_optimized


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    batch_size: int
    seq_len: int
    hidden_dim: int = 7168
    intermediate_dim: int = 2048
    num_experts: int = 256
    top_k: int = 8
    dtype: str = "float16"

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def flops(self) -> int:
        """Approximate FLOPs for MoE forward pass."""
        # Per token: 3 GEMMs (gate, up, down) x 2 (mul-add)
        # gate/up: hidden_dim x intermediate_dim
        # down: intermediate_dim x hidden_dim
        per_token = 2 * self.hidden_dim * self.intermediate_dim * 2  # gate + up
        per_token += 2 * self.intermediate_dim * self.hidden_dim     # down
        per_token *= self.top_k  # Each token goes to top_k experts
        return self.num_tokens * per_token


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    kernel_name: str
    latency_ms: float
    tokens_per_sec: float
    tflops: float
    memory_mb: float
    valid: bool
    error: Optional[str] = None


def benchmark_kernel(
    forward_fn,
    config: BenchmarkConfig,
    num_warmup: int = 10,
    num_iter: int = 50,
) -> BenchmarkResult:
    """Run benchmark for a single kernel configuration."""
    device = torch.device('cuda')
    dtype = getattr(torch, config.dtype)

    try:
        # Create inputs
        hidden_states = torch.randn(
            config.batch_size, config.seq_len, config.hidden_dim,
            device=device, dtype=dtype
        )
        router_logits = torch.randn(
            config.batch_size, config.seq_len, config.num_experts,
            device=device, dtype=dtype
        )
        expert_weights_gate = torch.randn(
            config.num_experts, config.hidden_dim, config.intermediate_dim,
            device=device, dtype=dtype
        )
        expert_weights_up = torch.randn(
            config.num_experts, config.hidden_dim, config.intermediate_dim,
            device=device, dtype=dtype
        )
        expert_weights_down = torch.randn(
            config.num_experts, config.intermediate_dim, config.hidden_dim,
            device=device, dtype=dtype
        )

        # Memory before
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        # Warmup
        for _ in range(num_warmup):
            _ = forward_fn(
                hidden_states, router_logits,
                expert_weights_gate, expert_weights_up, expert_weights_down,
                top_k=config.top_k
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iter):
            _ = forward_fn(
                hidden_states, router_logits,
                expert_weights_gate, expert_weights_up, expert_weights_down,
                top_k=config.top_k
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iter

        # Memory after
        mem_peak = torch.cuda.max_memory_allocated()
        memory_mb = (mem_peak - mem_before) / 1024 / 1024

        latency_ms = elapsed * 1000
        tokens_per_sec = config.num_tokens / elapsed
        tflops = config.flops / elapsed / 1e12

        return BenchmarkResult(
            config=config,
            kernel_name=forward_fn.__name__,
            latency_ms=latency_ms,
            tokens_per_sec=tokens_per_sec,
            tflops=tflops,
            memory_mb=memory_mb,
            valid=True,
        )

    except Exception as e:
        return BenchmarkResult(
            config=config,
            kernel_name=forward_fn.__name__,
            latency_ms=0,
            tokens_per_sec=0,
            tflops=0,
            memory_mb=0,
            valid=False,
            error=str(e),
        )


def run_benchmarks(
    configs: List[BenchmarkConfig],
    kernels: Dict[str, callable],
    output_file: Optional[str] = None,
) -> List[BenchmarkResult]:
    """Run all benchmarks across configurations and kernels."""
    results = []

    print(f"\n{'='*80}")
    print(f"MoE KERNEL BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"{'='*80}\n")

    header = f"{'Config':<25} {'Kernel':<20} {'Latency (ms)':<15} {'Tok/s':<12} {'TFLOPS':<10} {'Memory (MB)':<12}"
    print(header)
    print("-" * len(header))

    for config in configs:
        for name, kernel_fn in kernels.items():
            result = benchmark_kernel(kernel_fn, config)
            result.kernel_name = name
            results.append(result)

            if result.valid:
                config_str = f"B{config.batch_size}S{config.seq_len}E{config.num_experts}"
                print(f"{config_str:<25} {name:<20} {result.latency_ms:<15.2f} "
                      f"{result.tokens_per_sec:<12.0f} {result.tflops:<10.2f} {result.memory_mb:<12.1f}")
            else:
                print(f"{config_str:<25} {name:<20} FAILED: {result.error[:40]}")

    # Summary: speedups
    print(f"\n{'='*80}")
    print("SPEEDUP SUMMARY (vs baseline)")
    print(f"{'='*80}")

    baseline_results = {
        (r.config.batch_size, r.config.seq_len): r
        for r in results if r.kernel_name == "baseline" and r.valid
    }

    for result in results:
        if result.kernel_name == "baseline" or not result.valid:
            continue
        key = (result.config.batch_size, result.config.seq_len)
        if key in baseline_results:
            baseline = baseline_results[key]
            speedup = baseline.latency_ms / result.latency_ms
            config_str = f"B{result.config.batch_size}S{result.config.seq_len}"
            print(f"{config_str}: {result.kernel_name} = {speedup:.2f}x")

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    return results


# DeepSeek-V3 style configurations
DEEPSEEK_V3_CONFIGS = [
    # Decode: small batch, single token
    BenchmarkConfig(batch_size=1, seq_len=1, num_experts=256, top_k=8),
    BenchmarkConfig(batch_size=8, seq_len=1, num_experts=256, top_k=8),
    BenchmarkConfig(batch_size=32, seq_len=1, num_experts=256, top_k=8),

    # Prefill: larger batches
    BenchmarkConfig(batch_size=1, seq_len=128, num_experts=256, top_k=8),
    BenchmarkConfig(batch_size=1, seq_len=512, num_experts=256, top_k=8),
    BenchmarkConfig(batch_size=1, seq_len=2048, num_experts=256, top_k=8),
    BenchmarkConfig(batch_size=8, seq_len=128, num_experts=256, top_k=8),
    BenchmarkConfig(batch_size=8, seq_len=512, num_experts=256, top_k=8),
]

# Smaller configs for T4 testing
T4_TEST_CONFIGS = [
    BenchmarkConfig(batch_size=1, seq_len=1, hidden_dim=2048, intermediate_dim=512, num_experts=64, top_k=4),
    BenchmarkConfig(batch_size=8, seq_len=1, hidden_dim=2048, intermediate_dim=512, num_experts=64, top_k=4),
    BenchmarkConfig(batch_size=1, seq_len=128, hidden_dim=2048, intermediate_dim=512, num_experts=64, top_k=4),
    BenchmarkConfig(batch_size=8, seq_len=128, hidden_dim=2048, intermediate_dim=512, num_experts=64, top_k=4),
]

KERNELS = {
    "baseline": moe_forward_baseline,
    "colmajor": moe_forward_colmajor,
    "splitk": moe_forward_splitk,
    "padding_free": moe_forward_padding_free,
    "optimized": moe_forward_optimized,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full", "deepseek"], default="test")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "test":
        configs = T4_TEST_CONFIGS
    elif args.mode == "deepseek":
        configs = DEEPSEEK_V3_CONFIGS
    else:
        configs = T4_TEST_CONFIGS + DEEPSEEK_V3_CONFIGS

    run_benchmarks(configs, KERNELS, args.output)
