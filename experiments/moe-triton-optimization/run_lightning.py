#!/usr/bin/env python3
"""
Run MoE benchmark on Lightning.ai T4 instance.

Usage:
    python run_lightning.py
"""

import subprocess
import sys
import os

# Lightning.ai Studio API
LIGHTNING_USER_ID = os.environ.get("LIGHTNING_USER_ID", "")
LIGHTNING_API_KEY = os.environ.get("LIGHTNING_API_KEY", "")

# Script to run on Lightning
REMOTE_SCRIPT = '''
#!/bin/bash
set -e

echo "=== Setting up environment ==="
pip install torch triton --quiet

echo "=== Creating MoE kernel files ==="
mkdir -p moe_kernels

# baseline_moe.py
cat > moe_kernels/baseline_moe.py << 'BASELINE_EOF'
"""Baseline MoE Kernel"""
import torch
import triton
import triton.language as tl

def moe_forward_baseline(
    hidden_states,
    router_logits,
    expert_weights_gate,
    expert_weights_up,
    expert_weights_down,
    top_k=8,
):
    batch, seq, hidden_dim = hidden_states.shape
    num_experts = router_logits.shape[-1]
    intermediate_dim = expert_weights_gate.shape[-1]

    num_tokens = batch * seq
    hidden_flat = hidden_states.view(num_tokens, hidden_dim)
    router_flat = router_logits.view(num_tokens, num_experts)

    routing_weights, selected_experts = torch.topk(router_flat, top_k, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)

    output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    for expert_idx in range(num_experts):
        expert_mask = (selected_experts == expert_idx).any(dim=-1)
        if not expert_mask.any():
            continue

        token_indices = expert_mask.nonzero(as_tuple=True)[0]
        expert_input = hidden_flat[token_indices]

        expert_positions = (selected_experts[token_indices] == expert_idx).float()
        weights = (routing_weights[token_indices] * expert_positions).sum(dim=-1, keepdim=True)

        gate_out = torch.mm(expert_input, expert_weights_gate[expert_idx])
        up_out = torch.mm(expert_input, expert_weights_up[expert_idx])
        hidden = torch.nn.functional.silu(gate_out) * up_out
        expert_output = torch.mm(hidden, expert_weights_down[expert_idx])

        output[token_indices] += weights * expert_output

    return output.view(batch, seq, hidden_dim)
BASELINE_EOF

# optimized_moe.py (padding-free with token sorting)
cat > moe_kernels/optimized_moe.py << 'OPTIMIZED_EOF'
"""Optimized MoE Kernel with padding-free token buffers"""
import torch
import triton
import triton.language as tl

class OptimizedTokenBuffer:
    def __init__(self, router_logits, top_k):
        self.num_tokens, self.num_experts = router_logits.shape
        self.top_k = top_k
        self.device = router_logits.device

        topk_weights, topk_indices = torch.topk(router_logits, top_k, dim=-1)
        self.routing_weights = torch.softmax(topk_weights, dim=-1)

        token_ids = torch.arange(self.num_tokens, device=self.device)
        self.expanded_token_ids = token_ids.unsqueeze(-1).expand(-1, top_k).reshape(-1)
        self.expanded_expert_ids = topk_indices.reshape(-1)
        self.expanded_weights = self.routing_weights.reshape(-1)

        sort_indices = torch.argsort(self.expanded_expert_ids, stable=True)
        self.sorted_token_ids = self.expanded_token_ids[sort_indices]
        self.sorted_expert_ids = self.expanded_expert_ids[sort_indices]
        self.sorted_weights = self.expanded_weights[sort_indices]

        self.tokens_per_expert = torch.bincount(
            self.sorted_expert_ids, minlength=self.num_experts
        )
        self.expert_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(self.tokens_per_expert, dim=0)
        ])
        self.total_sorted_tokens = self.sorted_token_ids.shape[0]


def moe_forward_optimized(
    hidden_states,
    router_logits,
    expert_weights_gate,
    expert_weights_up,
    expert_weights_down,
    top_k=8,
):
    batch, seq, hidden_dim = hidden_states.shape
    num_experts = router_logits.shape[-1]
    intermediate_dim = expert_weights_gate.shape[-1]

    num_tokens = batch * seq
    hidden_flat = hidden_states.view(num_tokens, hidden_dim)
    router_flat = router_logits.view(num_tokens, num_experts)

    token_buffer = OptimizedTokenBuffer(router_flat, top_k)
    sorted_hidden = hidden_flat[token_buffer.sorted_token_ids]
    output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    for expert_idx in range(num_experts):
        num_expert_tokens = token_buffer.tokens_per_expert[expert_idx].item()
        if num_expert_tokens == 0:
            continue

        start = token_buffer.expert_offsets[expert_idx].item()
        end = token_buffer.expert_offsets[expert_idx + 1].item()

        expert_input = sorted_hidden[start:end]
        expert_token_ids = token_buffer.sorted_token_ids[start:end]
        expert_weights = token_buffer.sorted_weights[start:end]

        gate_out = torch.mm(expert_input, expert_weights_gate[expert_idx])
        up_out = torch.mm(expert_input, expert_weights_up[expert_idx])
        hidden = torch.nn.functional.silu(gate_out) * up_out
        expert_output = torch.mm(hidden, expert_weights_down[expert_idx])

        output.index_add_(0, expert_token_ids, expert_weights.unsqueeze(-1) * expert_output)

    return output.view(batch, seq, hidden_dim)
OPTIMIZED_EOF

# Test script
cat > test_moe.py << 'TEST_EOF'
import torch
import time
import sys
sys.path.insert(0, 'moe_kernels')

from baseline_moe import moe_forward_baseline
from optimized_moe import moe_forward_optimized

print("=" * 60)
print("MoE KERNEL BENCHMARK")
print(f"Device: {torch.cuda.get_device_name()}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

device = torch.device('cuda')
dtype = torch.float16

# Correctness test
print("\\nCorrectness test...")
hidden = torch.randn(2, 16, 256, device=device, dtype=dtype)
router = torch.randn(2, 16, 32, device=device, dtype=dtype)
gate_w = torch.randn(32, 256, 128, device=device, dtype=dtype)
up_w = torch.randn(32, 256, 128, device=device, dtype=dtype)
down_w = torch.randn(32, 128, 256, device=device, dtype=dtype)

baseline_out = moe_forward_baseline(hidden, router, gate_w, up_w, down_w, top_k=4)
optimized_out = moe_forward_optimized(hidden, router, gate_w, up_w, down_w, top_k=4)

diff = (baseline_out - optimized_out).abs().max().item()
print(f"Max diff: {diff:.6f}")
if diff < 0.1:
    print("Correctness: PASSED")
else:
    print("Correctness: FAILED")
    sys.exit(1)

# Benchmark
print("\\nPerformance benchmark...")
configs = [
    # Decode
    {"batch": 1, "seq": 1, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
    {"batch": 8, "seq": 1, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
    {"batch": 32, "seq": 1, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
    # Prefill
    {"batch": 1, "seq": 64, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
    {"batch": 1, "seq": 256, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
    {"batch": 4, "seq": 128, "hidden": 1024, "intermediate": 512, "experts": 64, "top_k": 4},
]

print(f"{'Config':<30} {'Baseline (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
print("-" * 70)

def benchmark(fn, hidden, router, gate_w, up_w, down_w, top_k, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn(hidden, router, gate_w, up_w, down_w, top_k=top_k)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(hidden, router, gate_w, up_w, down_w, top_k=top_k)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000

for cfg in configs:
    hidden = torch.randn(cfg['batch'], cfg['seq'], cfg['hidden'], device=device, dtype=dtype)
    router = torch.randn(cfg['batch'], cfg['seq'], cfg['experts'], device=device, dtype=dtype)
    gate_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
    up_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
    down_w = torch.randn(cfg['experts'], cfg['intermediate'], cfg['hidden'], device=device, dtype=dtype)

    baseline_time = benchmark(moe_forward_baseline, hidden, router, gate_w, up_w, down_w, cfg['top_k'])
    optimized_time = benchmark(moe_forward_optimized, hidden, router, gate_w, up_w, down_w, cfg['top_k'])

    speedup = baseline_time / optimized_time
    config_str = f"B{cfg['batch']}S{cfg['seq']}E{cfg['experts']}K{cfg['top_k']}"
    print(f"{config_str:<30} {baseline_time:<15.2f} {optimized_time:<15.2f} {speedup:.2f}x")

print("\\n" + "=" * 60)
print("BENCHMARK COMPLETE")
TEST_EOF

echo "=== Running benchmark ==="
python test_moe.py
'''


def run_on_lightning():
    """Run the benchmark on Lightning.ai."""
    import tempfile
    import shutil

    # Check if lightning CLI is available
    try:
        result = subprocess.run(
            ["lightning", "--version"],
            capture_output=True, text=True
        )
        print(f"Lightning CLI: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Lightning CLI not found. Install with: pip install lightning-sdk")
        print("\nFalling back to local execution...")
        run_local()
        return

    # For now, just run locally
    print("Running locally (Lightning.ai integration coming soon)...")
    run_local()


def run_local():
    """Run the test locally."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check CUDA
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Running on: {torch.cuda.get_device_name()}")

    # Run test
    result = subprocess.run(
        [sys.executable, "test_moe.py", "--bench"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode


if __name__ == "__main__":
    run_local()
