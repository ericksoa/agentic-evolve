#!/usr/bin/env python3
"""
MoE Kernel Benchmark for H100/H200
Self-contained script for Lightning.ai execution
"""

import torch
import torch.nn.functional as F
import time
import sys

print("=" * 70)
print("MoE KERNEL BENCHMARK - H100")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 70)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

# ============================================================================
# BASELINE MoE KERNEL
# ============================================================================

def moe_forward_baseline(
    hidden_states,
    router_logits,
    expert_weights_gate,
    expert_weights_up,
    expert_weights_down,
    top_k=8,
):
    """Baseline MoE - naive expert loop with padding overhead."""
    batch, seq, hidden_dim = hidden_states.shape
    num_experts = router_logits.shape[-1]

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
        hidden = F.silu(gate_out) * up_out
        expert_output = torch.mm(hidden, expert_weights_down[expert_idx])

        output[token_indices] += weights * expert_output

    return output.view(batch, seq, hidden_dim)


# ============================================================================
# OPTIMIZED MoE KERNEL (Padding-Free + Column-Major Style)
# ============================================================================

class OptimizedTokenBuffer:
    """Padding-free token buffer with expert sorting."""

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


def moe_forward_optimized(
    hidden_states,
    router_logits,
    expert_weights_gate,
    expert_weights_up,
    expert_weights_down,
    top_k=8,
):
    """
    Optimized MoE with:
    1. Padding-free token buffers (no wasted compute)
    2. Token sorting by expert (coalesced memory access)
    3. index_add_ for efficient scatter
    """
    batch, seq, hidden_dim = hidden_states.shape
    num_experts = router_logits.shape[-1]

    num_tokens = batch * seq
    hidden_flat = hidden_states.view(num_tokens, hidden_dim)
    router_flat = router_logits.view(num_tokens, num_experts)

    # Create padding-free token buffer
    token_buffer = OptimizedTokenBuffer(router_flat, top_k)

    # Gather sorted hidden states (coalesced read)
    sorted_hidden = hidden_flat[token_buffer.sorted_token_ids]

    output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    # Process each expert with exact token count (no padding!)
    for expert_idx in range(num_experts):
        num_expert_tokens = token_buffer.tokens_per_expert[expert_idx].item()
        if num_expert_tokens == 0:
            continue

        start = token_buffer.expert_offsets[expert_idx].item()
        end = token_buffer.expert_offsets[expert_idx + 1].item()

        expert_input = sorted_hidden[start:end]
        expert_token_ids = token_buffer.sorted_token_ids[start:end]
        expert_weights = token_buffer.sorted_weights[start:end]

        # Fused gate+up with SwiGLU
        gate_out = torch.mm(expert_input, expert_weights_gate[expert_idx])
        up_out = torch.mm(expert_input, expert_weights_up[expert_idx])
        hidden = F.silu(gate_out) * up_out
        expert_output = torch.mm(hidden, expert_weights_down[expert_idx])

        # Efficient scatter-add
        output.index_add_(0, expert_token_ids, expert_weights.unsqueeze(-1) * expert_output)

    return output.view(batch, seq, hidden_dim)


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark(fn, hidden, router, gate_w, up_w, down_w, top_k, warmup=20, iters=100):
    """Benchmark with proper warmup."""
    # Warmup
    for _ in range(warmup):
        _ = fn(hidden, router, gate_w, up_w, down_w, top_k=top_k)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(hidden, router, gate_w, up_w, down_w, top_k=top_k)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    device = torch.device('cuda')
    dtype = torch.float16

    # ========== CORRECTNESS TEST ==========
    print("\n[1/2] CORRECTNESS TEST")
    print("-" * 50)

    hidden = torch.randn(2, 32, 512, device=device, dtype=dtype)
    router = torch.randn(2, 32, 64, device=device, dtype=dtype)
    gate_w = torch.randn(64, 512, 256, device=device, dtype=dtype)
    up_w = torch.randn(64, 512, 256, device=device, dtype=dtype)
    down_w = torch.randn(64, 256, 512, device=device, dtype=dtype)

    baseline_out = moe_forward_baseline(hidden, router, gate_w, up_w, down_w, top_k=4)
    optimized_out = moe_forward_optimized(hidden, router, gate_w, up_w, down_w, top_k=4)

    diff = (baseline_out - optimized_out).abs().max().item()
    rel_diff = diff / (baseline_out.abs().mean().item() + 1e-6)
    print(f"Max absolute diff: {diff:.6f}")
    print(f"Relative diff: {rel_diff:.6f}")

    if diff < 0.1:
        print("Correctness: PASSED ✓")
    else:
        print("Correctness: FAILED ✗")
        sys.exit(1)

    # ========== PERFORMANCE BENCHMARK ==========
    print("\n[2/2] PERFORMANCE BENCHMARK")
    print("-" * 50)

    # DeepSeek-V3 style configurations
    configs = [
        # Decode (single token generation - latency critical)
        {"batch": 1, "seq": 1, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Decode B1"},
        {"batch": 8, "seq": 1, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Decode B8"},
        {"batch": 32, "seq": 1, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Decode B32"},

        # Prefill (prompt processing - throughput critical)
        {"batch": 1, "seq": 128, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Prefill 128"},
        {"batch": 1, "seq": 512, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Prefill 512"},
        {"batch": 1, "seq": 2048, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Prefill 2K"},

        # Batch prefill
        {"batch": 4, "seq": 512, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Batch 4x512"},
        {"batch": 8, "seq": 256, "hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8, "name": "Batch 8x256"},
    ]

    print(f"\n{'Config':<20} {'Tokens':<10} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10} {'Tok/s':<12}")
    print("=" * 80)

    results = []
    for cfg in configs:
        try:
            hidden = torch.randn(cfg['batch'], cfg['seq'], cfg['hidden'], device=device, dtype=dtype)
            router = torch.randn(cfg['batch'], cfg['seq'], cfg['experts'], device=device, dtype=dtype)
            gate_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
            up_w = torch.randn(cfg['experts'], cfg['hidden'], cfg['intermediate'], device=device, dtype=dtype)
            down_w = torch.randn(cfg['experts'], cfg['intermediate'], cfg['hidden'], device=device, dtype=dtype)

            tokens = cfg['batch'] * cfg['seq']

            baseline_time = benchmark(moe_forward_baseline, hidden, router, gate_w, up_w, down_w, cfg['top_k'])
            optimized_time = benchmark(moe_forward_optimized, hidden, router, gate_w, up_w, down_w, cfg['top_k'])

            speedup = baseline_time / optimized_time
            tps = tokens / (optimized_time / 1000)

            results.append({
                'name': cfg['name'],
                'tokens': tokens,
                'baseline': baseline_time,
                'optimized': optimized_time,
                'speedup': speedup,
                'tps': tps
            })

            print(f"{cfg['name']:<20} {tokens:<10} {baseline_time:<12.2f} {optimized_time:<12.2f} {speedup:<10.2f}x {tps:<12.0f}")

        except Exception as e:
            print(f"{cfg['name']:<20} ERROR: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        max_speedup = max(r['speedup'] for r in results)
        min_speedup = min(r['speedup'] for r in results)

        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Best speedup:    {max_speedup:.2f}x ({[r['name'] for r in results if r['speedup'] == max_speedup][0]})")
        print(f"Worst speedup:   {min_speedup:.2f}x ({[r['name'] for r in results if r['speedup'] == min_speedup][0]})")

        # Decode vs Prefill analysis
        decode_results = [r for r in results if 'Decode' in r['name']]
        prefill_results = [r for r in results if 'Prefill' in r['name'] or 'Batch' in r['name']]

        if decode_results:
            decode_avg = sum(r['speedup'] for r in decode_results) / len(decode_results)
            print(f"\nDecode avg speedup:  {decode_avg:.2f}x")
        if prefill_results:
            prefill_avg = sum(r['speedup'] for r in prefill_results) / len(prefill_results)
            print(f"Prefill avg speedup: {prefill_avg:.2f}x")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
