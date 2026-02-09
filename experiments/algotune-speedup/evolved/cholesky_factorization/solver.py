"""
Approach: Blocked Cholesky with Numba + scipy LAPACK fallback
Mutation of gen0_a: Replace pure scipy approach with a blocked (tiled) Cholesky
decomposition in Numba for small-to-medium matrices. The blocked algorithm
processes NB columns at a time, which keeps the working set in L1/L2 cache.
Within each block, we use a column-oriented inner loop with reciprocal
multiplication. For large matrices, fall back to scipy LAPACK.
Key difference from other mutations: tiled/blocked approach for cache optimization.
"""
import numpy as np
import numba as nb
import scipy.linalg


@nb.njit(cache=False, fastmath=True)
def _cholesky_blocked(A, L, n, nb_size):
    """Blocked Cholesky decomposition for better cache utilization.

    Processes nb_size columns at a time. Within each block:
    1. Factor the diagonal block (small Cholesky)
    2. Solve for the panel below the diagonal block
    This keeps memory accesses localized for better cache performance.
    """
    for jb in range(0, n, nb_size):
        jend = min(jb + nb_size, n)

        # Step 1: Factor the diagonal block [jb:jend, jb:jend]
        # This is a small Cholesky on the block, column-oriented
        for j in range(jb, jend):
            # Diagonal element
            s = A[j, j]
            # Subtract contributions from previous blocks (columns 0..jb-1)
            for k in range(jb):
                s -= L[j, k] * L[j, k]
            # Subtract contributions from current block (columns jb..j-1)
            for k in range(jb, j):
                s -= L[j, k] * L[j, k]
            ljj = np.sqrt(s)
            L[j, j] = ljj
            inv_ljj = 1.0 / ljj

            # Sub-diagonal elements in the panel below
            for i in range(j + 1, n):
                s2 = A[i, j]
                # Contributions from previous blocks
                for k in range(jb):
                    s2 -= L[i, k] * L[j, k]
                # Contributions from current block
                for k in range(jb, j):
                    s2 -= L[i, k] * L[j, k]
                L[i, j] = s2 * inv_ljj


@nb.njit(cache=False, fastmath=True)
def _cholesky_simple(A, L, n):
    """Simple column-oriented Cholesky for very small matrices."""
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        ljj = np.sqrt(s)
        L[j, j] = ljj
        inv_ljj = 1.0 / ljj
        for i in range(j + 1, n):
            s2 = A[i, j]
            for k in range(j):
                s2 -= L[i, k] * L[j, k]
            L[i, j] = s2 * inv_ljj


class Solver:
    # Below this threshold, use numba JIT
    NUMBA_THRESHOLD = 80
    # Block size for tiled Cholesky (tuned for L1 cache ~32KB)
    BLOCK_SIZE = 16
    # Below this, use simple (non-blocked) Cholesky
    SIMPLE_THRESHOLD = 24

    def __init__(self):
        # Warm up both JIT functions
        _warmup_A = np.array([[4.0, 2.0], [2.0, 3.0]])
        _warmup_L = np.zeros((2, 2))
        _cholesky_simple(_warmup_A, _warmup_L, 2)
        _warmup_L2 = np.zeros((2, 2))
        _cholesky_blocked(_warmup_A, _warmup_L2, 2, 2)

    def solve(self, problem):
        A = problem["matrix"]
        n = A.shape[0]
        if n <= self.SIMPLE_THRESHOLD:
            # Very small: simple column-oriented Cholesky
            L = np.zeros((n, n), dtype=np.float64)
            _cholesky_simple(A, L, n)
        elif n <= self.NUMBA_THRESHOLD:
            # Medium: blocked Cholesky for cache efficiency
            L = np.zeros((n, n), dtype=np.float64)
            _cholesky_blocked(A, L, n, self.BLOCK_SIZE)
        else:
            # Large: scipy LAPACK is fastest
            A_work = A.copy(order='F')
            L = scipy.linalg.cholesky(A_work, lower=True, overwrite_a=True, check_finite=False)
        return {"Cholesky": {"L": L}}
