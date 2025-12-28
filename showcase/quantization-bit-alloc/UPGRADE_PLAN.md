# Upgrade Plan: Real Quantization with GPT-2

**Goal**: Replace synthetic sensitivity curves with real quantization evaluation using GPT-2 small on CPU (Apple M2).

**Status**: PLANNED (execute when compute available)

---

## Overview

Current state: Synthetic benchmark with artificial sensitivity curves and fitness cliffs.

Target state: Real quantization of GPT-2 small (124M params) with perplexity-based fitness, no artificial thresholds.

---

## Phase 1: Python Evaluator

### 1.1 Create `python/eval_gpt2.py`

```python
#!/usr/bin/env python3
"""
Real quantization evaluator for GPT-2 small.

Applies a bit allocation plan to GPT-2 and measures perplexity.
Outputs JSON with perplexity, model size, and bit histogram.

Usage:
    python eval_gpt2.py --plan allocation.json --mode fast
    python eval_gpt2.py --plan allocation.json --mode verify
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Fixed seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Eval corpus paths (committed to repo)
FAST_CORPUS = Path(__file__).parent.parent / "data" / "eval_fast.txt"
VERIFY_CORPUS = Path(__file__).parent.parent / "data" / "eval_verify.txt"

# Model constants
MODEL_NAME = "gpt2"  # 124M params
MAX_LENGTH = 256
FAST_TOKENS = 2048
VERIFY_TOKENS = 10240

# Bit width sizes (bytes per param)
BIT_SIZES = {
    "fp32": 4,
    "fp16": 2,
    "int8": 1,
    "int4": 0.5,
}


def load_model_and_tokenizer():
    """Load GPT-2 small from local cache or download once."""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def quantize_to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to INT8 (weight-only, symmetric)."""
    scale = tensor.abs().max() / 127.0
    if scale == 0:
        return tensor
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return (quantized.float() * scale).to(tensor.dtype)


def quantize_to_int4(tensor: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Quantize tensor to INT4 (weight-only, group-wise)."""
    original_shape = tensor.shape
    tensor_flat = tensor.view(-1)

    # Pad to multiple of group_size
    pad_len = (group_size - len(tensor_flat) % group_size) % group_size
    if pad_len > 0:
        tensor_flat = torch.nn.functional.pad(tensor_flat, (0, pad_len))

    tensor_groups = tensor_flat.view(-1, group_size)
    scales = tensor_groups.abs().max(dim=1, keepdim=True).values / 7.0
    scales = scales.clamp(min=1e-8)

    quantized = torch.round(tensor_groups / scales).clamp(-8, 7)
    dequantized = (quantized * scales).view(-1)

    # Remove padding
    if pad_len > 0:
        dequantized = dequantized[:-pad_len]

    return dequantized.view(original_shape).to(tensor.dtype)


def apply_bit_allocation(model: GPT2LMHeadModel, allocation: Dict[str, str]) -> Dict[str, int]:
    """
    Apply bit allocation to model weights in-place.

    Args:
        model: GPT-2 model
        allocation: { layer_name: bit_width } where bit_width in ["fp32", "fp16", "int8", "int4"]

    Returns:
        Bit histogram { "fp32": n, "fp16": n, "int8": n, "int4": n }
    """
    histogram = {"fp32": 0, "fp16": 0, "int8": 0, "int4": 0}

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Find matching allocation (support partial matches)
            bit_width = "fp32"  # default
            for alloc_name, alloc_width in allocation.items():
                if alloc_name in name or name in alloc_name:
                    bit_width = alloc_width
                    break

            # Apply quantization
            if bit_width == "fp32":
                param.data = param.data.float()
            elif bit_width == "fp16":
                param.data = param.data.half().float()  # Simulate FP16 precision
            elif bit_width == "int8":
                param.data = quantize_to_int8(param.data)
            elif bit_width == "int4":
                param.data = quantize_to_int4(param.data)

            histogram[bit_width] += param.numel()

    return histogram


def calculate_model_size(histogram: Dict[str, int]) -> int:
    """Calculate model size in bytes from bit histogram."""
    total_bytes = 0
    for bit_width, count in histogram.items():
        total_bytes += int(count * BIT_SIZES[bit_width])
    return total_bytes


def evaluate_perplexity(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    corpus_path: Path,
    max_tokens: int,
) -> float:
    """
    Evaluate perplexity on a text corpus.

    Uses sliding window with stride = max_length // 2 for efficiency.
    """
    with open(corpus_path, "r") as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0][:max_tokens]

    # Sliding window evaluation
    stride = MAX_LENGTH // 2
    nlls = []

    for i in range(0, len(input_ids) - 1, stride):
        begin = max(0, i - MAX_LENGTH + stride)
        end = min(i + stride, len(input_ids))

        target_begin = max(0, i)
        target_end = end

        input_chunk = input_ids[begin:end].unsqueeze(0)
        target_chunk = input_chunk.clone()

        # Mask tokens before target_begin
        if begin < target_begin:
            target_chunk[0, :target_begin - begin] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            nll = outputs.loss.item() * (target_end - target_begin)
            nlls.append(nll)

        if end >= len(input_ids):
            break

    # Calculate perplexity
    total_nll = sum(nlls)
    total_tokens = min(max_tokens, len(input_ids)) - 1
    perplexity = torch.exp(torch.tensor(total_nll / total_tokens)).item()

    return perplexity


def main():
    parser = argparse.ArgumentParser(description="GPT-2 quantization evaluator")
    parser.add_argument("--plan", type=str, required=True, help="Path to JSON bit allocation plan")
    parser.add_argument("--mode", choices=["fast", "verify"], default="fast")
    args = parser.parse_args()

    start_time = time.time()

    # Load allocation plan
    with open(args.plan) as f:
        allocation = json.load(f)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Apply quantization
    histogram = apply_bit_allocation(model, allocation)
    model_size = calculate_model_size(histogram)

    # Select corpus
    corpus_path = FAST_CORPUS if args.mode == "fast" else VERIFY_CORPUS
    max_tokens = FAST_TOKENS if args.mode == "fast" else VERIFY_TOKENS

    # Evaluate
    perplexity = evaluate_perplexity(model, tokenizer, corpus_path, max_tokens)

    eval_time = time.time() - start_time

    # Output result
    result = {
        "perplexity": perplexity,
        "model_size_bytes": model_size,
        "bit_histogram": histogram,
        "eval_time_seconds": eval_time,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

### 1.2 Create evaluation corpus

**File**: `data/eval_fast.txt` (~2k tokens)
- Use public domain text (e.g., excerpt from Project Gutenberg)
- Commit directly to repo

**File**: `data/eval_verify.txt` (~10k tokens)
- Different text slice (no overlap with fast)
- Same source for consistency

**Source options**:
- Pride and Prejudice (Gutenberg #1342)
- Alice in Wonderland (Gutenberg #11)
- Shakespeare sonnets

### 1.3 Dependencies

**File**: `python/requirements.txt`
```
torch>=2.0.0
transformers>=4.30.0
```

---

## Phase 2: Rust Evaluation Bridge

### 2.1 Create `rust/src/eval_bridge.rs`

```rust
//! Bridge between Rust evolution and Python evaluator.
//!
//! Emits bit allocation as JSON, invokes Python, parses results.

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};
use serde::{Deserialize, Serialize};
use crate::{BitWidth, LayerInfo};

/// Bit allocation plan for Python evaluator
#[derive(Serialize)]
pub struct AllocationPlan {
    pub allocations: HashMap<String, String>,
}

/// Result from Python evaluator
#[derive(Deserialize, Debug)]
pub struct EvalResult {
    pub perplexity: f64,
    pub model_size_bytes: u64,
    pub bit_histogram: HashMap<String, u64>,
    pub eval_time_seconds: f64,
}

impl AllocationPlan {
    /// Create plan from layer allocations
    pub fn from_allocations(allocations: &[(String, BitWidth)]) -> Self {
        let map: HashMap<String, String> = allocations
            .iter()
            .map(|(name, bw)| (name.clone(), bw.to_string()))
            .collect();
        Self { allocations: map }
    }

    /// Write plan to JSON file
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.allocations)?;
        std::fs::write(path, json)
    }
}

/// Evaluate a bit allocation using the Python evaluator
pub fn evaluate(
    plan: &AllocationPlan,
    mode: &str,  // "fast" or "verify"
    python_path: &str,
) -> Result<EvalResult, String> {
    // Write plan to temp file
    let plan_path = "/tmp/quant_plan.json";
    plan.write_to_file(plan_path)
        .map_err(|e| format!("Failed to write plan: {}", e))?;

    // Invoke Python evaluator
    let output = Command::new("python3")
        .arg(python_path)
        .arg("--plan")
        .arg(plan_path)
        .arg("--mode")
        .arg(mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run evaluator: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Evaluator failed: {}", stderr));
    }

    // Parse result
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse result: {} (output: {})", e, stdout))
}

impl std::fmt::Display for BitWidth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitWidth::INT4 => write!(f, "int4"),
            BitWidth::INT8 => write!(f, "int8"),
            BitWidth::FP16 => write!(f, "fp16"),
            BitWidth::FP32 => write!(f, "fp32"),
        }
    }
}
```

### 2.2 Update `rust/src/lib.rs`

Add GPT-2 layer definitions:

```rust
/// GPT-2 small layer names for bit allocation
pub const GPT2_LAYERS: &[&str] = &[
    // Embeddings
    "wte",      // token embeddings
    "wpe",      // position embeddings

    // Transformer blocks (0-11)
    "h.0.ln_1", "h.0.attn", "h.0.ln_2", "h.0.mlp",
    "h.1.ln_1", "h.1.attn", "h.1.ln_2", "h.1.mlp",
    "h.2.ln_1", "h.2.attn", "h.2.ln_2", "h.2.mlp",
    "h.3.ln_1", "h.3.attn", "h.3.ln_2", "h.3.mlp",
    "h.4.ln_1", "h.4.attn", "h.4.ln_2", "h.4.mlp",
    "h.5.ln_1", "h.5.attn", "h.5.ln_2", "h.5.mlp",
    "h.6.ln_1", "h.6.attn", "h.6.ln_2", "h.6.mlp",
    "h.7.ln_1", "h.7.attn", "h.7.ln_2", "h.7.mlp",
    "h.8.ln_1", "h.8.attn", "h.8.ln_2", "h.8.mlp",
    "h.9.ln_1", "h.9.attn", "h.9.ln_2", "h.9.mlp",
    "h.10.ln_1", "h.10.attn", "h.10.ln_2", "h.10.mlp",
    "h.11.ln_1", "h.11.attn", "h.11.ln_2", "h.11.mlp",

    // Final layer norm
    "ln_f",
];

/// Remove INT2 (not supported), keep only realistic bit widths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitWidth {
    INT4,
    INT8,
    FP16,
    FP32,
}
```

---

## Phase 3: New Fitness Function

### 3.1 Smooth fitness (no cliffs)

```rust
/// Calculate fitness from evaluation results.
///
/// Fitness = compression_ratio - λ * max(0, (P - P_baseline) / P_baseline)
///
/// Where:
/// - P = perplexity
/// - P_baseline = FP16 baseline perplexity
/// - λ = 10.0 (makes >10% perplexity degradation dominate compression)
pub fn calculate_fitness(
    result: &EvalResult,
    baseline_perplexity: f64,
    baseline_size: u64,
) -> f64 {
    const LAMBDA: f64 = 10.0;

    let compression_ratio = baseline_size as f64 / result.model_size_bytes as f64;
    let perplexity_degradation = (result.perplexity - baseline_perplexity) / baseline_perplexity;
    let penalty = LAMBDA * perplexity_degradation.max(0.0);

    (compression_ratio - penalty).max(0.0)
}
```

### 3.2 Validation requirements

- VERIFY perplexity must be within 2% of FAST perplexity
- If VERIFY degrades >2% vs FAST → strategy is INVALID
- Only valid strategies can become champions

---

## Phase 4: Real Baselines

### 4.1 Implement baseline strategies

```rust
/// All FP32 (unquantized baseline)
pub fn all_fp32() -> Vec<(String, BitWidth)> {
    GPT2_LAYERS.iter().map(|l| (l.to_string(), BitWidth::FP32)).collect()
}

/// All FP16 (standard mixed precision)
pub fn all_fp16() -> Vec<(String, BitWidth)> {
    GPT2_LAYERS.iter().map(|l| (l.to_string(), BitWidth::FP16)).collect()
}

/// Weight-only INT8
pub fn all_int8() -> Vec<(String, BitWidth)> {
    GPT2_LAYERS.iter().map(|l| (l.to_string(), BitWidth::INT8)).collect()
}

/// LayerNorm FP32 + rest INT8
pub fn layernorm_fp32_rest_int8() -> Vec<(String, BitWidth)> {
    GPT2_LAYERS.iter().map(|l| {
        let bw = if l.contains("ln_") || *l == "ln_f" {
            BitWidth::FP32
        } else {
            BitWidth::INT8
        };
        (l.to_string(), bw)
    }).collect()
}
```

---

## Phase 5: Evolution Integration

### 5.1 Modify `evolved.rs` interface

The `Evolved` struct now produces a full allocation plan:

```rust
impl BitAllocationHeuristic for Evolved {
    fn allocate_all(&self) -> Vec<(String, BitWidth)> {
        GPT2_LAYERS.iter().enumerate().map(|(idx, layer_name)| {
            let total = GPT2_LAYERS.len();
            let pos = idx as f64 / total as f64;

            let bw = if layer_name.contains("ln_") || *layer_name == "ln_f" {
                // LayerNorm always FP32
                BitWidth::FP32
            } else if pos < 0.075 || pos > 0.925 {
                // Edge protection
                BitWidth::FP32
            } else if layer_name.contains("attn") {
                // Attention layers - experiment with INT8
                BitWidth::INT8
            } else {
                // MLP layers - can handle more compression
                BitWidth::INT8
            };

            (layer_name.to_string(), bw)
        }).collect()
    }
}
```

### 5.2 Benchmark flow

```
1. Generate baseline metrics (all_fp16)
   - FAST perplexity
   - Model size

2. For each candidate strategy:
   a. Generate allocation plan (Rust)
   b. Write plan to JSON
   c. Invoke Python evaluator (FAST mode)
   d. Parse results
   e. Calculate fitness

3. Select generation winner
   a. Run VERIFY on winner
   b. Check perplexity consistency
   c. If valid, update champion

4. After evolution:
   a. Run full VERIFY on final champion
   b. Print detailed results
```

---

## Phase 6: File Structure

```
showcase/quantization-bit-alloc/
├── README.md                 # Updated docs
├── UPGRADE_PLAN.md          # This file
├── python/
│   ├── eval_gpt2.py         # Real evaluator
│   └── requirements.txt     # Python deps
├── data/
│   ├── eval_fast.txt        # ~2k tokens (committed)
│   └── eval_verify.txt      # ~10k tokens (committed)
├── rust/
│   ├── Cargo.toml           # Updated deps
│   ├── Cargo.lock
│   └── src/
│       ├── lib.rs           # GPT-2 layer defs
│       ├── baselines.rs     # Real baselines
│       ├── evolved.rs       # Champion strategy
│       ├── eval_bridge.rs   # Rust-Python bridge
│       └── benchmark.rs     # Updated benchmark
```

---

## Phase 7: Execution Checklist

### Setup (one-time)
- [ ] Install Python dependencies: `pip install -r python/requirements.txt`
- [ ] Download GPT-2 model: `python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"`
- [ ] Create evaluation corpus files
- [ ] Verify Python evaluator works standalone

### Testing
- [ ] Run baseline evaluation (all_fp16)
- [ ] Verify FAST eval completes in <3 seconds
- [ ] Verify VERIFY eval completes in <15 seconds
- [ ] Compare perplexities are consistent (<2% difference)

### Evolution
- [ ] Run single generation manually
- [ ] Verify fitness calculation
- [ ] Run full evolution (6+ generations)
- [ ] Validate final champion with VERIFY

### Validation
- [ ] Champion beats all baselines
- [ ] VERIFY perplexity within 2% of FAST
- [ ] Results are deterministic (run twice)
- [ ] No network access during eval

---

## Expected Results

### Baseline Performance (estimated)

| Strategy | Compression | Perplexity | Fitness |
|----------|-------------|------------|---------|
| all_fp32 | 1.0x | ~30 | 1.0 |
| all_fp16 | 2.0x | ~30 | 2.0 |
| all_int8 | 4.0x | ~32-35 | 3.5-4.0 |
| ln_fp32_rest_int8 | ~3.8x | ~31-32 | 3.5-3.8 |

### Evolution Target

Find strategies that achieve:
- **Compression**: 3-4x over FP32
- **Perplexity**: <5% degradation vs FP16
- **Fitness**: >3.5 (beating simple INT8)

Potential discoveries:
- Which attention layers tolerate INT4?
- Should early MLP layers stay higher precision?
- Does embedding quantization hurt more than MLP?

---

## Anti-Cheating Measures

1. **Separate eval corpora**: FAST and VERIFY use non-overlapping text
2. **Consistency check**: VERIFY must be within 2% of FAST
3. **No sensitivity proxies**: All metrics come from real forward passes
4. **Deterministic seeds**: Same results every run
5. **No gradient information**: Weight-only quantization, no fine-tuning

---

## Notes for Tomorrow

1. **Start with Python evaluator** - get it working standalone first
2. **Cache model download** - GPT-2 small is ~500MB, download once
3. **Test INT4 carefully** - group-wise quantization can be tricky
4. **Monitor memory** - M2 has limited RAM, keep batch size small
5. **Time budget**: Each FAST eval ~1-3s, each generation ~30s with 8 candidates

Good luck!
