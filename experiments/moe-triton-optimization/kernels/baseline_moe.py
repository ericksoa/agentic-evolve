"""
Baseline MoE Kernel - Standard Implementation

This implements a straightforward MoE forward pass for benchmarking.
Based on vLLM's approach but simplified for clarity.

MoE Forward Pass:
1. Router computes expert scores for each token
2. Top-K experts selected per token
3. Tokens dispatched to experts (with padding)
4. Expert FFN computation
5. Outputs combined with routing weights
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def moe_gemm_kernel(
    # Pointers
    A_ptr,  # Input activations [num_tokens, hidden_dim]
    B_ptr,  # Expert weights [num_experts, hidden_dim, intermediate_dim]
    C_ptr,  # Output [num_tokens, intermediate_dim]
    # Token-to-expert mapping
    expert_ids_ptr,  # Which expert each token goes to [num_tokens]
    token_ids_ptr,   # Original token indices [num_tokens]
    # Dimensions
    num_tokens,
    hidden_dim,
    intermediate_dim,
    num_experts,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Standard MoE GEMM kernel.

    Each program computes a BLOCK_M x BLOCK_N tile of the output.
    Tokens are pre-sorted by expert, so consecutive tokens use same weights.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Token range for this block
    token_start = pid_m * BLOCK_M
    token_offsets = token_start + tl.arange(0, BLOCK_M)
    token_mask = token_offsets < num_tokens

    # Load expert IDs for this token block
    expert_ids = tl.load(expert_ids_ptr + token_offsets, mask=token_mask, other=0)

    # Output column range
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < intermediate_dim

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, hidden_dim, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_dim

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + token_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = token_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N] - need to handle per-expert weights
        # For simplicity, assume all tokens in block use same expert (sorted)
        expert_id = tl.load(expert_ids_ptr + token_start)
        b_ptrs = B_ptr + expert_id * stride_be + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C_ptr + token_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def moe_forward_baseline(
    hidden_states: torch.Tensor,  # [batch, seq, hidden_dim]
    router_logits: torch.Tensor,  # [batch, seq, num_experts]
    expert_weights_gate: torch.Tensor,  # [num_experts, hidden_dim, intermediate_dim]
    expert_weights_up: torch.Tensor,    # [num_experts, hidden_dim, intermediate_dim]
    expert_weights_down: torch.Tensor,  # [num_experts, intermediate_dim, hidden_dim]
    top_k: int = 8,
) -> torch.Tensor:
    """
    Baseline MoE forward pass.

    Uses standard PyTorch operations with padding for comparison.
    """
    batch, seq, hidden_dim = hidden_states.shape
    num_experts = router_logits.shape[-1]
    intermediate_dim = expert_weights_gate.shape[-1]

    # Flatten to [num_tokens, hidden_dim]
    num_tokens = batch * seq
    hidden_flat = hidden_states.view(num_tokens, hidden_dim)
    router_flat = router_logits.view(num_tokens, num_experts)

    # Compute routing weights (softmax over top-k experts)
    routing_weights, selected_experts = torch.topk(router_flat, top_k, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)

    # Initialize output
    output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    # Process each expert (naive loop - baseline)
    for expert_idx in range(num_experts):
        # Find tokens routed to this expert
        expert_mask = (selected_experts == expert_idx).any(dim=-1)
        if not expert_mask.any():
            continue

        token_indices = expert_mask.nonzero(as_tuple=True)[0]
        expert_input = hidden_flat[token_indices]

        # Get routing weight for this expert
        expert_positions = (selected_experts[token_indices] == expert_idx).float()
        weights = (routing_weights[token_indices] * expert_positions).sum(dim=-1, keepdim=True)

        # Expert computation: gate * up -> activation -> down
        gate_out = torch.mm(expert_input, expert_weights_gate[expert_idx])
        up_out = torch.mm(expert_input, expert_weights_up[expert_idx])

        # SwiGLU activation
        hidden = torch.nn.functional.silu(gate_out) * up_out

        # Down projection
        expert_output = torch.mm(hidden, expert_weights_down[expert_idx])

        # Accumulate weighted output
        output[token_indices] += weights * expert_output

    return output.view(batch, seq, hidden_dim)


def benchmark_baseline(
    batch_size: int = 8,
    seq_len: int = 2048,
    hidden_dim: int = 7168,
    intermediate_dim: int = 2048,
    num_experts: int = 256,
    top_k: int = 8,
    num_warmup: int = 10,
    num_iter: int = 50,
) -> float:
    """Benchmark baseline MoE implementation."""
    import time

    device = torch.device('cuda')
    dtype = torch.float16

    # Create inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    router_logits = torch.randn(batch_size, seq_len, num_experts, device=device, dtype=dtype)

    # Create expert weights
    expert_weights_gate = torch.randn(num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype)
    expert_weights_up = torch.randn(num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype)
    expert_weights_down = torch.randn(num_experts, intermediate_dim, hidden_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        _ = moe_forward_baseline(
            hidden_states, router_logits,
            expert_weights_gate, expert_weights_up, expert_weights_down,
            top_k=top_k
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iter):
        _ = moe_forward_baseline(
            hidden_states, router_logits,
            expert_weights_gate, expert_weights_up, expert_weights_down,
            top_k=top_k
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iter * 1000  # ms

    return elapsed


if __name__ == "__main__":
    print("Baseline MoE Benchmark")
    print("=" * 50)

    # DeepSeek-V3 style configuration
    configs = [
        {"batch_size": 1, "seq_len": 1, "num_experts": 256, "top_k": 8},
        {"batch_size": 1, "seq_len": 128, "num_experts": 256, "top_k": 8},
        {"batch_size": 8, "seq_len": 128, "num_experts": 256, "top_k": 8},
        {"batch_size": 8, "seq_len": 2048, "num_experts": 256, "top_k": 8},
    ]

    for cfg in configs:
        try:
            latency = benchmark_baseline(**cfg)
            tokens = cfg["batch_size"] * cfg["seq_len"]
            print(f"B={cfg['batch_size']}, S={cfg['seq_len']}, E={cfg['num_experts']}, K={cfg['top_k']}: "
                  f"{latency:.2f}ms ({tokens/latency*1000:.0f} tok/s)")
        except Exception as e:
            print(f"Config {cfg} failed: {e}")
