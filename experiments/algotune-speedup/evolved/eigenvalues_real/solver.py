"""
Gen2 Variant C: dsyev with zero-copy Fortran layout via symmetric transpose trick.

Mutation from gen0_d (dsyev only, 2.81x): Cache optimization via zero-copy array layout.

Key insight: For symmetric matrices, A == A^T. If the input is C-contiguous (which
np.random.randn produces), then A.T is F-contiguous WITHOUT any data copy - it's just
a different view of the same memory buffer. Since the matrix is symmetric, A.T has
the same values as A. This eliminates the expensive np.asfortranarray copy that
dominates for small-to-medium matrices.

Additional optimizations:
- Analytical fast paths for n=1 and n=2 (avoid LAPACK call overhead)
- Explicit lwork=max(3*n-1,1) to avoid internal lwork calculation
- Pre-bound function refs as default args
- list(reversed(w)) vs w[::-1].tolist() comparison

Dispatch:
- n=1: Direct return
- n=2: Closed-form quadratic eigenvalues
- n>=3: dsyev with symmetric transpose trick for zero-copy F-contiguous
"""
import numpy as np
from scipy.linalg.lapack import dsyev
import math

_dsyev = dsyev
_sqrt = math.sqrt


class Solver:
    __slots__ = ()

    def __init__(self):
        # Pre-warm dsyev
        _w2 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64, order='F')
        _dsyev(_w2, compute_v=0, lower=1, overwrite_a=1)

    def solve(self, problem, _dsyev=_dsyev, _sqrt=_sqrt):
        n = problem.shape[0]

        if n == 1:
            return [float(problem[0, 0])]

        if n == 2:
            a = problem[0, 0]
            b = problem[0, 1]
            d = problem[1, 1]
            trace = a + d
            disc = _sqrt(max((a - d) * (a - d) + 4.0 * b * b, 0.0))
            return [(trace + disc) * 0.5, (trace - disc) * 0.5]

        # Zero-copy Fortran layout for symmetric matrices:
        # For symmetric A, A == A.T. If A is C-contiguous, A.T is F-contiguous
        # as a view (no copy). Since the matrix is symmetric, the data is identical.
        # This avoids the expensive np.asfortranarray copy entirely.
        if problem.flags['F_CONTIGUOUS']:
            a = problem
        else:
            # problem.T is F-contiguous if problem is C-contiguous
            # For symmetric matrices, problem.T contains the same values
            a = problem.T

        w, _, info = _dsyev(a, compute_v=0, lower=1, lwork=max(3 * n - 1, 1), overwrite_a=1)
        return w[::-1].tolist()
