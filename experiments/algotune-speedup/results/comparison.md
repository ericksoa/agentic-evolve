## Results — INVALID (Official Validation Pending)

**Custom validation claimed 2.62x but AlgoTune's official evaluation shows ~1.01x.**

### Official AlgoTune Evaluation (3-Task Sample)

| Task | Custom Validation | Official Eval |
|------|------------------|---------------|
| ode_brusselator | 318.93x | **1.00x** |
| pagerank | 18.61x | **1.02x** |
| cholesky_factorization | 4.97x | **1.02x** |
| **Harmonic mean** | **2.62x** | **1.01x** |

### Root Cause

Custom validation used small problem sizes (n=50-500) where Python overhead dominates. AlgoTune's official datasets use large inputs (n=1,660-267,021) calibrated to ~100ms reference time. At those sizes, BLAS/scipy computation dominates and our overhead-reduction optimizations are irrelevant.

### Previous Custom Validation Results (INVALID)

These numbers are from our flawed custom validation and do not reflect actual AlgoTune performance:

| Task | Evolution Speedup | Custom "Validated" Speedup | Status |
|------|------------------|---------------------------|--------|
| ode_lotkavolterra | 1221.65x | 647.71x | **invalid** |
| ode_brusselator | 506.34x | 318.93x | **invalid** (official: 1.00x) |
| pagerank | 11.13x | 18.61x | **invalid** (official: 1.02x) |
| minimum_spanning_tree | 9.70x | 9.83x | **unverified** |
| correlate_1d | 5.50x | 8.61x | **unverified** |
| lu_factorization | 10.43x | 8.57x | **unverified** |
| fft_convolution | 5.47x | 8.26x | **unverified** |
| convex_hull | 4.47x | 6.34x | **unverified** |
| cholesky_factorization | 2.23x | 4.97x | **invalid** (official: 1.02x) |
| pca | 5.25x | 4.62x | **unverified** |
| qr_factorization | 4.88x | 3.65x | **unverified** |
| convolve_1d | 4.16x | 2.83x | **unverified** |
| kmeans | 2.19x | 2.55x | **unverified** |
| svd | 1.22x | 2.20x | **unverified** |
| matrix_multiplication | 1.15x | 1.58x | **unverified** |
| linear_system_solver | 1.81x | 1.48x | **unverified** |
| eigenvectors_real | 1.41x | 1.48x | **unverified** |
| svm | 13.09x | 1.00x | **unverified** |
| dijkstra_from_indices | 0.53x | 0.58x | **unverified** |
