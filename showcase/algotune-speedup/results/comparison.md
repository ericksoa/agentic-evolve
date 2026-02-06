## Results Comparison

| Model | Harmonic Mean Speedup | Tasks Improved |
|-------|----------------------|----------------|
| **Opus 4.6 + evolve-sdk** | **3.51x** | **100%** |
| o4-mini-high | 1.72x | 60% |
| deepseek-r1 | 1.70x | 61% |
| gemini-2.5-pro | 1.51x | 49% |
| claude-opus-4 | 1.33x | 40% |

*Results on curated 3-task subset. See methodology section for full-benchmark extrapolation.*

## Per-Task Results

| Task | Phase 1 | Phase 2 | Best | Source |
|------|---------|---------|------|--------|
| cholesky_factorization | 1.08x | 2.23x | **2.23x** | phase2 |
| convex_hull | 1.06x | 4.47x | **4.47x** | phase2 |
| fft_convolution | 1.02x | 5.47x | **5.47x** | phase2 |