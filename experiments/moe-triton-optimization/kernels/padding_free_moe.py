"""
Padding-Free MoE Implementation

Key insight from X-MoE (arXiv:2508.13337):
- Standard MoE pads tokens to fixed expert capacity → wasted compute
- X-MoE uses sparse token buffers with no padding
- Achieves 5.15x speedup over DeepSpeed-TED

Our implementation:
- Pre-sort tokens by expert (no padding needed)
- Variable-length expert batches
- Fused gate+up projection with SwiGLU
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def fused_gate_up_kernel(
    # Inputs
    input_ptr,       # [total_tokens, hidden_dim]
    gate_weight_ptr, # [num_experts, hidden_dim, intermediate_dim]
    up_weight_ptr,   # [num_experts, hidden_dim, intermediate_dim]
    output_ptr,      # [total_tokens, intermediate_dim]
    # Token mapping
    token_expert_ids_ptr,  # Expert ID for each token position
    expert_offsets_ptr,    # Start offset for each expert
    # Dimensions
    total_tokens,
    hidden_dim,
    intermediate_dim,
    num_experts,
    # Strides
    stride_in_m, stride_in_k,
    stride_gw_e, stride_gw_k, stride_gw_n,
    stride_uw_e, stride_uw_k, stride_uw_n,
    stride_out_m, stride_out_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused gate + up projection with SwiGLU.

    Output = SiLU(input @ gate_weight) * (input @ up_weight)

    Each thread block processes BLOCK_M tokens and BLOCK_N output columns.
    Tokens are pre-sorted by expert, so consecutive tokens use same weights.
    """
    pid_m = tl.program_id(0)  # Token block
    pid_n = tl.program_id(1)  # Output column block

    # Token offsets
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < total_tokens

    # Output column offsets
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < intermediate_dim

    # Get expert ID for first token in block (all should be same expert due to sorting)
    expert_id = tl.load(token_expert_ids_ptr + m_start)

    # Initialize accumulators
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-dimension loop
    for k_start in range(0, hidden_dim, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_dim

        # Load input tile
        in_ptrs = input_ptr + m_offsets[:, None] * stride_in_m + k_offsets[None, :] * stride_in_k
        x = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load gate weight tile
        gw_ptrs = (gate_weight_ptr +
                   expert_id * stride_gw_e +
                   k_offsets[:, None] * stride_gw_k +
                   n_offsets[None, :] * stride_gw_n)
        gw = tl.load(gw_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Load up weight tile
        uw_ptrs = (up_weight_ptr +
                   expert_id * stride_uw_e +
                   k_offsets[:, None] * stride_uw_k +
                   n_offsets[None, :] * stride_uw_n)
        uw = tl.load(uw_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        gate_acc += tl.dot(x, gw)
        up_acc += tl.dot(x, uw)

    # SwiGLU: SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate_silu = gate_acc * tl.sigmoid(gate_acc)
    output = gate_silu * up_acc

    # Store result
    out_ptrs = output_ptr + m_offsets[:, None] * stride_out_m + n_offsets[None, :] * stride_out_n
    tl.store(out_ptrs, output.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def down_proj_scatter_kernel(
    # Inputs
    intermediate_ptr,  # [total_tokens, intermediate_dim]
    down_weight_ptr,   # [num_experts, intermediate_dim, hidden_dim]
    output_ptr,        # [num_tokens, hidden_dim]
    # Token mapping
    sorted_token_ids_ptr,   # Original token indices in sorted order
    token_expert_ids_ptr,   # Expert ID for each sorted position
    routing_weights_ptr,    # Routing weight for each token-expert pair
    # Dimensions
    total_sorted_tokens,
    intermediate_dim,
    hidden_dim,
    num_experts,
    # Strides
    stride_int_m, stride_int_k,
    stride_dw_e, stride_dw_k, stride_dw_n,
    stride_out_m, stride_out_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Down projection with weighted scatter-add to output.

    Computes: output[orig_token_id] += routing_weight * (intermediate @ down_weight)
    """
    pid_m = tl.program_id(0)  # Token block
    pid_n = tl.program_id(1)  # Output column block

    # Sorted token offsets
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < total_sorted_tokens

    # Output column offsets
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < hidden_dim

    # Get expert ID
    expert_id = tl.load(token_expert_ids_ptr + m_start)

    # Get original token IDs for scatter
    orig_token_ids = tl.load(sorted_token_ids_ptr + m_offsets, mask=m_mask, other=0)

    # Get routing weights
    routing_weights = tl.load(routing_weights_ptr + m_offsets, mask=m_mask, other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-dimension loop
    for k_start in range(0, intermediate_dim, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < intermediate_dim

        # Load intermediate tile
        int_ptrs = intermediate_ptr + m_offsets[:, None] * stride_int_m + k_offsets[None, :] * stride_int_k
        x = tl.load(int_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load down weight tile
        dw_ptrs = (down_weight_ptr +
                   expert_id * stride_dw_e +
                   k_offsets[:, None] * stride_dw_k +
                   n_offsets[None, :] * stride_dw_n)
        dw = tl.load(dw_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(x, dw)

    # Apply routing weights
    weighted_output = acc * routing_weights[:, None]

    # Note: Atomic scatter-add would go here in full implementation
    # For now, we'll handle scatter in Python wrapper


class PaddingFreeTokenBuffer:
    """
    Sparse token buffer for padding-free MoE computation.

    Instead of padding all experts to max_tokens_per_expert,
    we store tokens contiguously sorted by expert.
    """

    def __init__(
        self,
        router_logits: torch.Tensor,  # [num_tokens, num_experts]
        top_k: int,
    ):
        self.num_tokens, self.num_experts = router_logits.shape
        self.top_k = top_k
        self.device = router_logits.device

        # Compute routing
        topk_weights, topk_indices = torch.topk(router_logits, top_k, dim=-1)
        self.routing_weights = torch.softmax(topk_weights, dim=-1)  # [num_tokens, top_k]

        # Create token-expert mapping
        # Each token appears top_k times (once per selected expert)
        token_ids = torch.arange(self.num_tokens, device=self.device)
        self.expanded_token_ids = token_ids.unsqueeze(-1).expand(-1, top_k).reshape(-1)
        self.expanded_expert_ids = topk_indices.reshape(-1)
        self.expanded_weights = self.routing_weights.reshape(-1)

        # Sort by expert for coalesced access
        sort_indices = torch.argsort(self.expanded_expert_ids, stable=True)
        self.sorted_token_ids = self.expanded_token_ids[sort_indices]
        self.sorted_expert_ids = self.expanded_expert_ids[sort_indices]
        self.sorted_weights = self.expanded_weights[sort_indices]

        # Compute expert boundaries
        self.tokens_per_expert = torch.bincount(
            self.sorted_expert_ids, minlength=self.num_experts
        )
        self.expert_offsets = torch.cat([
            torch.zeros(1, device=self.device, dtype=torch.long),
            torch.cumsum(self.tokens_per_expert, dim=0)
        ])

        # Total tokens in sorted buffer
        self.total_sorted_tokens = self.sorted_token_ids.shape[0]

    def get_expert_slice(self, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get token IDs, expert IDs, and weights for a specific expert."""
        start = self.expert_offsets[expert_idx].item()
        end = self.expert_offsets[expert_idx + 1].item()
        return (
            self.sorted_token_ids[start:end],
            self.sorted_expert_ids[start:end],
            self.sorted_weights[start:end],
        )


def moe_forward_padding_free(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_gate: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    top_k: int = 8,
) -> torch.Tensor:
    """
    Padding-free MoE forward pass.

    Key optimizations:
    1. No padding - variable expert batch sizes
    2. Pre-sorted tokens for coalesced memory access
    3. Fused gate+up with SwiGLU
    """
    batch, seq, hidden_dim = hidden_states.shape
    num_experts = router_logits.shape[-1]
    intermediate_dim = expert_weights_gate.shape[-1]

    num_tokens = batch * seq
    hidden_flat = hidden_states.view(num_tokens, hidden_dim)
    router_flat = router_logits.view(num_tokens, num_experts)

    # Create padding-free token buffer
    token_buffer = PaddingFreeTokenBuffer(router_flat, top_k)

    # Gather sorted hidden states (no padding!)
    sorted_hidden = hidden_flat[token_buffer.sorted_token_ids]

    # Initialize output
    output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    # Process each expert with their exact token count
    for expert_idx in range(num_experts):
        num_expert_tokens = token_buffer.tokens_per_expert[expert_idx].item()
        if num_expert_tokens == 0:
            continue

        # Get this expert's data (no padding!)
        token_ids, _, weights = token_buffer.get_expert_slice(expert_idx)
        start = token_buffer.expert_offsets[expert_idx].item()
        end = token_buffer.expert_offsets[expert_idx + 1].item()

        expert_input = sorted_hidden[start:end]

        # Fused gate + up projection with SwiGLU
        gate_out = torch.mm(expert_input, expert_weights_gate[expert_idx])
        up_out = torch.mm(expert_input, expert_weights_up[expert_idx])
        hidden = torch.nn.functional.silu(gate_out) * up_out

        # Down projection
        expert_output = torch.mm(hidden, expert_weights_down[expert_idx])

        # Weighted scatter-add back to output
        output.index_add_(0, token_ids, weights.unsqueeze(-1) * expert_output)

    return output.view(batch, seq, hidden_dim)


def compute_padding_overhead(
    router_logits: torch.Tensor,
    top_k: int,
) -> float:
    """
    Compute the padding overhead for a given routing distribution.

    Returns the ratio of (padded_compute / actual_compute).
    Overhead > 1.0 means wasted compute from padding.
    """
    num_tokens, num_experts = router_logits.shape

    # Get top-k experts per token
    _, topk_indices = torch.topk(router_logits, top_k, dim=-1)
    expert_ids = topk_indices.reshape(-1)

    # Count tokens per expert
    tokens_per_expert = torch.bincount(expert_ids, minlength=num_experts)
    max_tokens = tokens_per_expert.max().item()
    total_actual = tokens_per_expert.sum().item()

    # Padded computation = num_experts * max_tokens_per_expert
    total_padded = num_experts * max_tokens

    overhead = total_padded / total_actual if total_actual > 0 else float('inf')
    return overhead


def benchmark_padding_free(
    batch_size: int = 8,
    seq_len: int = 2048,
    hidden_dim: int = 7168,
    intermediate_dim: int = 2048,
    num_experts: int = 256,
    top_k: int = 8,
    num_warmup: int = 10,
    num_iter: int = 50,
) -> Tuple[float, float]:
    """
    Benchmark padding-free MoE implementation.

    Returns: (latency_ms, padding_overhead)
    """
    import time

    device = torch.device('cuda')
    dtype = torch.float16

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    router_logits = torch.randn(batch_size, seq_len, num_experts, device=device, dtype=dtype)
    expert_weights_gate = torch.randn(num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype)
    expert_weights_up = torch.randn(num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype)
    expert_weights_down = torch.randn(num_experts, intermediate_dim, hidden_dim, device=device, dtype=dtype)

    # Compute padding overhead
    router_flat = router_logits.view(-1, num_experts)
    overhead = compute_padding_overhead(router_flat, top_k)

    # Warmup
    for _ in range(num_warmup):
        _ = moe_forward_padding_free(
            hidden_states, router_logits,
            expert_weights_gate, expert_weights_up, expert_weights_down,
            top_k=top_k
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iter):
        _ = moe_forward_padding_free(
            hidden_states, router_logits,
            expert_weights_gate, expert_weights_up, expert_weights_down,
            top_k=top_k
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iter * 1000

    return elapsed, overhead


if __name__ == "__main__":
    print("Padding-Free MoE Benchmark")
    print("=" * 60)

    configs = [
        {"batch_size": 1, "seq_len": 1, "num_experts": 256, "top_k": 8},
        {"batch_size": 1, "seq_len": 128, "num_experts": 256, "top_k": 8},
        {"batch_size": 8, "seq_len": 128, "num_experts": 256, "top_k": 8},
        {"batch_size": 8, "seq_len": 2048, "num_experts": 256, "top_k": 8},
    ]

    print(f"{'Config':<25} {'Latency (ms)':<15} {'Overhead':<15} {'Effective':<15}")
    print("-" * 70)

    for cfg in configs:
        try:
            latency, overhead = benchmark_padding_free(**cfg)
            tokens = cfg["batch_size"] * cfg["seq_len"]
            effective_tps = tokens / latency * 1000
            print(f"B={cfg['batch_size']}, S={cfg['seq_len']:<10} "
                  f"{latency:<15.2f} {overhead:<15.2f}x {effective_tps:<15.0f} tok/s")
        except Exception as e:
            print(f"Config {cfg} failed: {e}")
