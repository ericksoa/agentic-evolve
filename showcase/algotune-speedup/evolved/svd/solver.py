"""
Gen1 Crossover: Hybrid combining size-dispatched fast paths (D) + PyTorch backend (E) +
eigendecomposition for rectangular matrices (F).

Multi-tier dispatch strategy:
  1. n=0: Pre-allocated empty arrays (from D)
  2. k=1: Analytical 1x1/1xm/nx1 SVD (from D)
  3. Highly rectangular (aspect > 1.5, k >= 30): eigh approach (from F)
  4. Small-medium square-ish (k <= 200): Direct LAPACK dgesdd (from D)
  5. Large square-ish (k > 200): PyTorch SVD if available (from E), else scipy.linalg.svd (from D)

Lazy proxy for V = Vh.T on all non-analytical paths.
"""
from typing import Any
import numpy as np

try:
    from scipy.linalg.lapack import dgesdd as _dgesdd
except ImportError:
    _dgesdd = None

try:
    from scipy.linalg import svd as _scipy_svd
except ImportError:
    _scipy_svd = None

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Pre-allocated constants
_E01 = np.zeros((0,), dtype=np.float64)

# Thresholds
_LARGE_THRESHOLD = 200
_ASPECT_THRESHOLD = 1.5
_MIN_SIZE_FOR_EIGH = 30


class _LazyTranspose:
    """Lazy Vh.T - defers transpose until np.array() is called."""
    __slots__ = ('_Vh',)

    def __init__(self, Vh):
        self._Vh = Vh

    def __array__(self, dtype=None):
        V = self._Vh.T
        if dtype is not None:
            V = V.astype(dtype, copy=False)
        return V


class _LazyTorchTranspose:
    """Lazy transpose of a torch Vh tensor, converting to numpy on demand."""
    __slots__ = ('_Vh',)

    def __init__(self, Vh):
        self._Vh = Vh

    def __array__(self, dtype=None):
        V = self._Vh.T.numpy()
        if dtype is not None:
            V = V.astype(dtype, copy=False)
        return V


class Solver:
    def __init__(self):
        self._dgesdd = _dgesdd
        self._scipy_svd = _scipy_svd
        self._use_torch = _HAS_TORCH

        # Pre-warm LAPACK
        if _dgesdd is not None:
            try:
                _w = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=np.float64)
                _dgesdd(_w, compute_uv=1, full_matrices=0, overwrite_a=1)
            except Exception:
                pass

        # Pre-warm PyTorch SVD
        if _HAS_TORCH:
            try:
                _w = torch.randn(4, 3, dtype=torch.float64)
                torch.linalg.svd(_w, full_matrices=False)
            except Exception:
                self._use_torch = False

    def _svd_via_eigh_tall(self, A, n, m):
        """SVD of tall matrix (n > m) via eigh(A^T @ A)."""
        AtA = A.T @ A
        eigenvalues, V = np.linalg.eigh(AtA)
        # Reverse to get descending singular values
        eigenvalues = eigenvalues[::-1]
        V = V[:, ::-1]
        s = np.sqrt(np.maximum(eigenvalues, 0.0))
        inv_s = np.where(s > 1e-10, 1.0 / s, 0.0)
        U = A @ (V * inv_s[np.newaxis, :])
        return U, s, V

    def _svd_via_eigh_wide(self, A, n, m):
        """SVD of wide matrix (m > n) via eigh(A @ A^T)."""
        AAt = A @ A.T
        eigenvalues, U = np.linalg.eigh(AAt)
        eigenvalues = eigenvalues[::-1]
        U = U[:, ::-1]
        s = np.sqrt(np.maximum(eigenvalues, 0.0))
        inv_s = np.where(s > 1e-10, 1.0 / s, 0.0)
        V = A.T @ (U * inv_s[np.newaxis, :])
        return U, s, V

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem["matrix"]
        n, m = A.shape
        k = min(n, m)

        # === Tier 0: Empty matrix (from D) ===
        if k == 0:
            return {
                "U": np.empty((n, 0), dtype=np.float64),
                "S": _E01,
                "V": np.empty((m, 0), dtype=np.float64),
            }

        # === Tier 1: Analytical fast paths for k=1 (from D) ===
        if k == 1:
            if n == 1 and m == 1:
                val = A[0, 0]
                s_val = abs(val)
                u_sign = 1.0 if val >= 0 else -1.0
                return {
                    "U": np.array([[u_sign]]),
                    "S": np.array([s_val]),
                    "V": np.array([[1.0]]),
                }
            elif n == 1:
                row = A[0, :]
                norm = np.linalg.norm(row)
                if norm < 1e-15:
                    return {
                        "U": np.array([[1.0]]),
                        "S": np.array([0.0]),
                        "V": np.zeros((m, 1), dtype=np.float64),
                    }
                u_sign = 1.0
                V = (row / norm).reshape(m, 1)
                if V[0, 0] < 0:
                    V = -V
                    u_sign = -1.0
                return {
                    "U": np.array([[u_sign]]),
                    "S": np.array([norm]),
                    "V": V,
                }
            else:
                col = A[:, 0]
                norm = np.linalg.norm(col)
                if norm < 1e-15:
                    return {
                        "U": np.zeros((n, 1), dtype=np.float64),
                        "S": np.array([0.0]),
                        "V": np.array([[1.0]]),
                    }
                U = (col / norm).reshape(n, 1)
                v_sign = 1.0
                if U[0, 0] < 0:
                    U = -U
                    v_sign = -1.0
                return {
                    "U": U,
                    "S": np.array([norm]),
                    "V": np.array([[v_sign]]),
                }

        # === Tier 2: Eigendecomposition for highly rectangular matrices (from F) ===
        if k >= _MIN_SIZE_FOR_EIGH:
            aspect = max(n, m) / max(k, 1)
            if aspect >= _ASPECT_THRESHOLD:
                try:
                    if n >= m:
                        U, s, V = self._svd_via_eigh_tall(A, n, m)
                    else:
                        U, s, V = self._svd_via_eigh_wide(A, n, m)
                    return {"U": U, "S": s, "V": V}
                except Exception:
                    pass  # Fall through to direct SVD

        # === Tier 3: Large square-ish matrices - PyTorch (from E) or scipy (from D) ===
        if k > _LARGE_THRESHOLD:
            # Try PyTorch first for large matrices (from E)
            if self._use_torch:
                try:
                    A_torch = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float64))
                    U_t, s_t, Vh_t = torch.linalg.svd(A_torch, full_matrices=False)
                    U = U_t.numpy()
                    s = s_t.numpy()
                    return {"U": U, "S": s, "V": _LazyTorchTranspose(Vh_t)}
                except Exception:
                    pass

            # Fallback for large: scipy.linalg.svd (from D)
            if self._scipy_svd is not None:
                U, s, Vh = self._scipy_svd(
                    A, full_matrices=False, check_finite=False, overwrite_a=False
                )
                return {"U": U, "S": s, "V": _LazyTranspose(Vh)}

        # === Tier 4: Small-medium square-ish matrices - Direct LAPACK dgesdd (from D) ===
        dgesdd = self._dgesdd
        if dgesdd is not None:
            Af = np.array(A, dtype=np.float64, order='F')
            u, s, vt, info = dgesdd(Af, compute_uv=1, full_matrices=0, overwrite_a=1)
            if info == 0:
                return {"U": u, "S": s, "V": _LazyTranspose(vt)}

        # === Final fallback: numpy ===
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        return {"U": U, "S": s, "V": _LazyTranspose(Vh)}
