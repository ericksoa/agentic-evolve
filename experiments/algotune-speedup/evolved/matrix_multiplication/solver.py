"""
Gen2 Crossover X: Best-of-breed hybrid combining three parent strategies.

Parents:
  - gen0_e (fitness 1.2199): Zero-copy transpose trick with dgemm - THE core algorithm
  - gen1b (fitness 1.2007): Aggressive micro-optimizations (module-level caching,
    __slots__, default arg caching, size-based dispatch)
  - gen0_d (fitness 1.1876): PyTorch backend with size-based dispatch concept

Crossover strategy:
  - From gen0_e: The zero-copy transpose trick as the core computation backend.
    A.T and B.T are already Fortran-contiguous views (no copy), so dgemm with
    trans_a=1, trans_b=1 computes A @ B with zero-copy overhead.
  - From gen1b: Micro-optimization infrastructure - module-level function caching,
    __slots__ on Solver, all function references as default args to eliminate
    per-call dict lookups. Also the size-based dispatch concept (numpy @ for small).
  - From gen0_d: The insight that different backends excel at different sizes.
    But instead of PyTorch (which has too much per-call overhead), we use:
    - numpy @ for small matrices (fast C-level dispatch, no wrapper overhead)
    - dgemm + pre-allocated output buffer + overwrite_c for larger matrices

Key innovation over all parents:
  - Pre-allocated Fortran-contiguous output buffer with overwrite_c=1 (from gen1a
    insights). This avoids dgemm's internal allocation for the result matrix.
  - Threshold tuned on n*p product (covers both rectangular and square matrices
    better than just 'n'). Threshold at 400 (≈20×20) balances dgemm wrapper
    overhead (~5μs) vs computation time.
  - All micro-optimizations combined: module-level warmup with cleanup, cached
    function refs, __slots__, default arg binding.
"""
import numpy as np
from scipy.linalg.blas import dgemm as _dgemm

# Module-level pre-warming - ensures dgemm BLAS internals are initialized once
# (from gen0_e, refined with cleanup from gen1b style)
_warmup = np.array([[1.0]], order='F')
_dgemm(1.0, _warmup, _warmup)
del _warmup

# Cache all references at module level (from gen1b's caching strategy)
_float64 = np.float64
_np_array = np.array
_np_empty = np.empty


class Solver:
    __slots__ = ()  # From gen1b: eliminate instance __dict__ overhead

    def solve(self, problem: dict,
              _dgemm=_dgemm,
              _array=_np_array,
              _empty=_np_empty,
              _f64=_float64) -> np.ndarray:
        # Direct array creation - no ascontiguousarray needed (from gen1b insight:
        # np.array from list-of-lists is already C-contiguous)
        A = _array(problem["A"], dtype=_f64)  # C-contiguous (n, m)
        B = _array(problem["B"], dtype=_f64)  # C-contiguous (m, p)

        n = A.shape[0]
        p = B.shape[1]

        # Size-based dispatch (concept from gen1b/gen0_d, threshold tuned):
        # For small matrices, numpy @ has less dispatch overhead than dgemm wrapper.
        # dgemm Python wrapper costs ~5-6μs (arg parsing, type checking, validation).
        # n*p < 400 covers roughly 20×20 square or equivalent rectangular matrices.
        if n * p < 400:
            return A @ B

        # Larger matrices: dgemm with zero-copy transpose trick (from gen0_e)
        # + pre-allocated output buffer (from gen1a/gen1c insights)
        #
        # A.T is Fortran-contiguous (m, n) - zero copy view
        # B.T is Fortran-contiguous (p, m) - zero copy view
        # dgemm(1.0, A.T, B.T, trans_a=1, trans_b=1) = (A.T).T @ (B.T).T = A @ B
        #
        # Pre-allocate F-contiguous output and use overwrite_c=1 to write directly
        # into it, avoiding dgemm's internal allocation path.
        C = _empty((n, p), dtype=_f64, order='F')
        return _dgemm(1.0, A.T, B.T, beta=0.0, c=C,
                      trans_a=1, trans_b=1, overwrite_c=1)
