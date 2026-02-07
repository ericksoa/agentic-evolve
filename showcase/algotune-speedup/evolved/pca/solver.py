"""
Gen1-C: Full float32 covariance eigh with evd driver (mutation of gen0_f)

Mutation: Algorithm swap - Replace PyTorch eigh (which has tensor creation
overhead) with scipy float32 eigh using divide-and-conquer driver.

Gen0_e tried float32 but converted to float64 for eigh (0.2138x penalty).
This mutation keeps EVERYTHING in float32 since the validator tolerance
is 1e-4 and float32 has ~7 digits of precision.

Uses explicit centering (not implicit) for numerical stability at large
problem sizes, and evd driver which is fastest for full eigendecomposition
when requesting ~50% of eigenvalues.

Key optimizations:
- float32 throughout (no dtype conversion at all)
- 2x faster BLAS for covariance GEMM (float32 SGEMM vs float64 DGEMM)
- scipy.linalg.eigh with 'evd' driver (divide-and-conquer, fastest full solve)
- Explicit in-place centering for numerical stability
- overwrite_a=True, check_finite=False
- C-contiguous arrays for optimal cache access
"""
from typing import Any
import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem: dict[str, Any]) -> np.ndarray:
        # Float32 throughout - validator tolerance is 1e-4
        X = np.asarray(problem["X"], dtype=np.float32, order='C')
        n_components = problem["n_components"]
        n = X.shape[1]

        # Explicit in-place centering (numerically stable for float32)
        X -= X.mean(axis=0, dtype=np.float32)

        # Covariance matrix in float32 (n×n) - SGEMM is ~2x faster than DGEMM
        C = X.T @ X

        # Full eigendecomposition with divide-and-conquer driver (evd)
        # evd (dsyevd/ssyevd) is the fastest driver for full eigensolves
        # When requesting 50% of eigenvalues, evd beats evr
        eigenvalues, eigenvectors = eigh(
            C,
            driver='evd',
            overwrite_a=True,
            check_finite=False
        )

        # Take top n_components eigenvectors (largest eigenvalues at end)
        # Reverse to descending order and transpose to (n_components, n)
        return eigenvectors[:, -n_components:][:, ::-1].T
