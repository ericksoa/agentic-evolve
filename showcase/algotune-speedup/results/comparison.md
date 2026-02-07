## Results Comparison

| Model | Harmonic Mean Speedup | Tasks Improved |
|-------|----------------------|----------------|
| **Opus 4.6 + evolve-sdk** | **2.95x** | **95%** |
| o4-mini-high | 1.72x | 60% |
| deepseek-r1 | 1.70x | 61% |
| gemini-2.5-pro | 1.51x | 49% |
| claude-opus-4 | 1.33x | 40% |

*Results on curated 19-task subset. See methodology section for full-benchmark extrapolation.*

## Per-Task Results

| Task | Phase 1 | Phase 2 | Best | Source |
|------|---------|---------|------|--------|
| cholesky_factorization | 1.08x | 2.23x | **2.23x** | phase2 |
| convex_hull | 1.06x | 4.47x | **4.47x** | phase2 |
| convolve_1d | - | 4.16x | **4.16x** | phase2 |
| correlate_1d | - | 5.50x | **5.50x** | phase2 |
| dct_type_I_scipy_fftpack | - | - | **invalid** | phase1 |
| dijkstra_from_indices | - | 0.53x | **1.00x** | phase2 |
| eigenvalues_real | - | - | **invalid** | phase1 |
| eigenvectors_real | - | 1.41x | **1.41x** | phase2 |
| fft_cmplx_scipy_fftpack | - | - | **invalid** | phase1 |
| fft_convolution | 1.02x | 5.47x | **5.47x** | phase2 |
| kmeans | - | 2.19x | **2.19x** | phase2 |
| lasso | - | - | **invalid** | phase1 |
| linear_system_solver | - | 1.81x | **1.81x** | phase2 |
| lu_factorization | - | 10.43x | **10.43x** | phase2 |
| matrix_exponential | - | - | **invalid** | phase1 |
| matrix_multiplication | - | 1.15x | **1.15x** | phase2 |
| minimum_spanning_tree | - | 9.70x | **9.70x** | phase2 |
| ode_brusselator | - | 506.34x | **506.34x** | phase2 |
| ode_lotkavolterra | - | 1221.65x | **1221.65x** | phase2 |
| pagerank | - | 11.13x | **11.13x** | phase2 |
| pca | - | 5.25x | **5.25x** | phase2 |
| qr_factorization | - | 4.88x | **4.88x** | phase2 |
| shortest_path_dijkstra | - | - | **invalid** | phase1 |
| svd | - | 1.22x | **1.22x** | phase2 |
| svm | - | 13.09x | **13.09x** | phase2 |