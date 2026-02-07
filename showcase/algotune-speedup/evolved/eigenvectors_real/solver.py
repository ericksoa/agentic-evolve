"""
Gen1 Crossover: Best of D (dsyevd direct LAPACK) + E (size-dependent dispatch).

Approach:
- From D: dsyevd with optimal workspace, pre-warming, F-contiguous, cached refs
- From E: size-dependent algorithm dispatch (dsyev for small, dsyevd for large)
  - Refined cutoff: dsyev for 3 <= n <= 25 (QR is lower overhead for small matrices)
  - dsyevd for n > 25 (divide-and-conquer wins asymptotically)
- Analytical fast paths for n=1 and n=2
- Optimized output conversion: w[::-1].tolist() and v[:, ::-1].T.tolist()
- All function references cached via closure defaults to avoid dict lookups
"""
import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import dsyev as _dsyev_func, dsyevd as _dsyevd_func
import math

_sqrt = math.sqrt
_array = np.array


class Solver:
    def __init__(self):
        # Pre-warm both LAPACK routines to eliminate first-call JIT/init overhead
        _w = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64, order='F')
        _dsyev_func(_w.copy(order='F'), compute_v=1, lower=1, overwrite_a=1)
        _dsyevd_func(_w.copy(order='F'), compute_v=1, lower=1, overwrite_a=1)

    def solve(self, problem: NDArray,
              _dsyev=_dsyev_func, _dsyevd=_dsyevd_func,
              _sqrt=_sqrt, _np_array=_array) -> tuple[list[float], list[list[float]]]:
        n = problem.shape[0]

        if n == 1:
            return ([float(problem[0, 0])], [[1.0]])

        if n == 2:
            a = problem[0, 0]
            b = problem[0, 1]
            d = problem[1, 1]
            trace = a + d
            diff = a - d
            disc = _sqrt(max(diff * diff + 4.0 * b * b, 0.0))
            lam1 = (trace + disc) * 0.5
            lam2 = (trace - disc) * 0.5

            if abs(b) > 1e-15:
                v1x = b
                v1y = lam1 - a
                norm1 = _sqrt(v1x * v1x + v1y * v1y)
                inv_norm1 = 1.0 / norm1
                v1x *= inv_norm1
                v1y *= inv_norm1
                v2x = -v1y
                v2y = v1x
            else:
                if a >= d:
                    v1x, v1y = 1.0, 0.0
                    v2x, v2y = 0.0, 1.0
                else:
                    v1x, v1y = 0.0, 1.0
                    v2x, v2y = 1.0, 0.0

            return ([lam1, lam2], [[v1x, v1y], [v2x, v2y]])

        # F-contiguous copy for LAPACK (FORTRAN column-major)
        a_fort = _np_array(problem, dtype=np.float64, order='F')

        # Size-dependent dispatch (from parent E, refined cutoff)
        if n <= 25:
            # dsyev (QR iteration) - lower overhead for small-to-medium matrices
            lwork = max(3 * n - 1, 1)
            w, v, info = _dsyev(a_fort, compute_v=1, lower=1, lwork=lwork,
                                overwrite_a=1)
        else:
            # dsyevd (divide-and-conquer) - better asymptotic for larger matrices
            lwork = 1 + 6 * n + 2 * n * n
            liwork = 3 + 5 * n
            w, v, info = _dsyevd(a_fort, compute_v=1, lower=1, lwork=lwork,
                                 liwork=liwork, overwrite_a=1)

        return (w[::-1].tolist(), v[:, ::-1].T.tolist())
