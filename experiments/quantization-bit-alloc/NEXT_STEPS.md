# Next Steps: Validation & Extension Plan

Plan for testing on Lepton boxes with stable internet.

## Phase 1: Scale Validation (GPT-2 Family)

Test if early MLP sensitivity scales with model size.

| Model | Params | Layers | Expected Test Time |
|-------|--------|--------|-------------------|
| gpt2 | 124M | 12 | ~10 min (done) |
| distilgpt2 | 82M | 6 | ~5 min (done) |
| gpt2-medium | 355M | 24 | ~30 min |
| gpt2-large | 774M | 36 | ~1 hour |
| gpt2-xl | 1.5B | 48 | ~2 hours |

**Hypothesis**: Early MLP sensitivity should persist. The "early" threshold might scale (e.g., first 10% of layers rather than first 25%).

**Commands**:
```bash
# Create configs for each model size
# For gpt2-medium (24 layers): test h.1-6 MLP FP16 vs h.12-17 vs h.18-23

python python/eval_gpt2.py --plan medium_fp16.json --mode verify --model gpt2-medium
python python/eval_gpt2.py --plan medium_all_int8.json --mode verify --model gpt2-medium
python python/eval_gpt2.py --plan medium_early_mlp.json --mode verify --model gpt2-medium
python python/eval_gpt2.py --plan medium_middle_mlp.json --mode verify --model gpt2-medium
python python/eval_gpt2.py --plan medium_late_mlp.json --mode verify --model gpt2-medium
```

---

## Phase 2: Architecture Generalization

Test on non-GPT architectures to see if early MLP sensitivity is universal.

### Priority 1: LLaMA-style models
```
TinyLlama-1.1B (22 layers)
Llama-2-7B (32 layers) - needs GPU
```

### Priority 2: Other architectures
```
microsoft/phi-2 (32 layers, 2.7B)
mistralai/Mistral-7B-v0.1 (32 layers)
```

**Key question**: Do LLaMA-style models (with RMSNorm, SwiGLU) show the same pattern?

---

## Phase 3: Real Quantization (Not Simulated)

Current evaluator simulates quantization by rounding weights. For production:

### Option A: bitsandbytes
```python
import bitsandbytes as bnb
# Load with actual INT8 quantization
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    load_in_8bit=True,
    device_map="auto"
)
```

### Option B: GPTQ
```bash
pip install auto-gptq
# Actual 4-bit/8-bit with calibration
```

### Option C: llama.cpp quantization
```bash
# Export to GGUF format with mixed precision
python convert.py gpt2 --outtype f16
./quantize gpt2-f16.gguf gpt2-q8_0.gguf q8_0
```

**Goal**: Verify the pattern holds with real quantized inference.

---

## Phase 4: Inference Speed Benchmarks

Compression is only useful if it translates to speed. Test on Lepton boxes:

```python
import time

def benchmark_inference(model, tokenizer, prompt, n_tokens=100, n_runs=10):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=n_tokens)
        times.append(time.perf_counter() - start)

    return {
        "tokens_per_second": n_tokens / (sum(times) / n_runs),
        "latency_ms": (sum(times) / n_runs) * 1000
    }
```

**Metrics to collect**:
- Tokens/second (throughput)
- Time to first token (latency)
- Memory usage (peak GPU memory)
- Batch size scaling

---

## Phase 5: Production Dataset Validation

WikiText-2 is small and academic. Test on production-relevant data:

| Dataset | Domain | Size |
|---------|--------|------|
| C4 | Web text | Large |
| The Pile | Mixed | Large |
| RedPajama | Training mix | Large |
| Code (StarCoder) | Programming | Medium |

**Goal**: Ensure pattern isn't WikiText-specific.

---

## Phase 6: Lepton-Specific Tests

If using Lepton AI infrastructure:

### A. Multi-GPU scaling
```python
# Test if quantization pattern works with tensor parallelism
# Lepton supports automatic model sharding
```

### B. Batch size optimization
```python
# Find optimal batch size for mixed-precision model
# Compare: all-FP16 vs all-INT8 vs champion
```

### C. Cost analysis
```
Calculate: $/1M tokens for each configuration
- FP16 baseline
- All INT8
- Champion (early MLP FP16)
```

---

## Quick Reference: Config Templates

### GPT-2 Medium (24 layers)
```json
{
  "_model": "gpt2-medium",
  "wte": "int8", "wpe": "int8",
  "h.0.attn": "int8", "h.0.mlp": "int8",
  "h.1.attn": "int8", "h.1.mlp": "fp16",
  "h.2.attn": "int8", "h.2.mlp": "fp16",
  "h.3.attn": "int8", "h.3.mlp": "fp16",
  "h.4.attn": "int8", "h.4.mlp": "fp16",
  "h.5.attn": "int8", "h.5.mlp": "fp16",
  "h.6.attn": "int8", "h.6.mlp": "fp16",
  "h.7-h.23": "int8",
  "ln_f": "fp32"
}
```

### Scaling rule (hypothesis)
```
FP16 MLP layers = first ceil(n_layers * 0.25) layers, excluding layer 0
Example:
- 6 layers:  h.0-1 (2 layers = 33%)
- 12 layers: h.1-3 (3 layers = 25%)
- 24 layers: h.1-6 (6 layers = 25%)
- 36 layers: h.1-9 (9 layers = 25%)
```

---

## Success Criteria

| Test | Pass Condition |
|------|----------------|
| GPT-2 medium | Early MLP > 2x better than middle/late |
| GPT-2 large | Pattern persists with scaling |
| LLaMA | Pattern generalizes to different arch |
| Real INT8 | No accuracy loss vs simulated |
| Speed | Champion faster than FP16 baseline |
| Production data | Pattern holds on non-WikiText |

---

## Estimated Timeline

| Phase | Effort | Hardware Needed |
|-------|--------|-----------------|
| 1. GPT-2 scaling | 2-4 hours | 1x GPU (16GB+) |
| 2. Architecture | 4-8 hours | 1x GPU (24GB+) |
| 3. Real quant | 2-4 hours | 1x GPU |
| 4. Speed bench | 2-4 hours | Lepton box |
| 5. Prod data | 4-8 hours | 1x GPU |
| 6. Lepton tests | 4-8 hours | Lepton boxes |

**Total**: ~20-40 hours of compute time

---

## Files to Create

When ready to run:
```bash
# Generate all config files
python scripts/generate_scaling_configs.py

# Run full validation suite
./scripts/run_validation.sh
```
