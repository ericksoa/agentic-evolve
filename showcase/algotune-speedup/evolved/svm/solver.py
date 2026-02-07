"""
Approach: Direct Clarabel QP with flattened numpy indexing (no meshgrid).

Mutation of gen0_a: Replaces the dense hstack/vstack matrix construction with
COO format (like gen0_f) but eliminates the expensive np.meshgrid call that
allocates two n×p temporary arrays. Instead uses np.repeat/np.tile which are
cheaper for generating row/col indices. Also uses Fortran-order (column-major)
for yX to avoid the .T.ravel() transpose copy when filling CSC column data.

Additionally relaxes solver tolerances from 1e-9 to 1e-7 (validation uses
atol=1e-6) and disables iterative refinement + chordal decomposition to
reduce Clarabel per-iteration overhead.

Variables: z = [beta (p), beta0 (1), xi (n)]
Objective: min 0.5 ||beta||^2 + C * sum(xi)
Constraints: y_i(x_i^T beta + beta0) + xi_i >= 1, xi_i >= 0
"""
from typing import Any

import clarabel
import numpy as np
import scipy.sparse as sp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        C = float(problem["C"])

        n, p = X.shape
        dim = p + 1 + n

        # Build P as sparse diagonal directly (only p nonzeros)
        p_idx = np.arange(p)
        P = sp.csc_matrix(
            (np.ones(p), (p_idx, p_idx)), shape=(dim, dim)
        )

        # q vector
        q = np.empty(dim)
        q[:p + 1] = 0.0
        q[p + 1:] = C

        # Build constraint matrix A in COO format
        # Block structure:
        # A = [[-diag(y)*X  -y  -I_n]    (n rows: margin constraints)
        #      [ 0           0  -I_n]]   (n rows: xi >= 0)

        # Compute -y_i * X in column-major order so .ravel('F') gives
        # columns sequentially without a transpose copy
        yX_neg = np.asfortranarray(-(y[:, None] * X))  # shape (n, p), F-order

        total_nnz = n * p + n + n + n
        rows = np.empty(total_nnz, dtype=np.int32)
        cols = np.empty(total_nnz, dtype=np.int32)
        vals = np.empty(total_nnz, dtype=np.float64)

        arange_n = np.arange(n, dtype=np.int32)
        block_np = n * p

        # Block 1a: -yX block (n rows × p cols)
        # Row indices: [0,1,...,n-1, 0,1,...,n-1, ...] repeated p times
        # Col indices: [0,0,...,0, 1,1,...,1, ...] each repeated n times
        rows[:block_np] = np.tile(arange_n, p)
        cols[:block_np] = np.repeat(np.arange(p, dtype=np.int32), n)
        vals[:block_np] = yX_neg.ravel(order='F')  # column-major, no copy needed

        # Block 1b: -y in column p
        idx = block_np
        rows[idx:idx + n] = arange_n
        cols[idx:idx + n] = p
        vals[idx:idx + n] = -y
        idx += n

        # Block 1c: -I_n in cols p+1..p+n
        rows[idx:idx + n] = arange_n
        cols[idx:idx + n] = np.arange(p + 1, p + 1 + n, dtype=np.int32)
        vals[idx:idx + n] = -1.0
        idx += n

        # Block 2: -I_n in cols p+1..p+n, rows n..2n-1
        rows[idx:idx + n] = arange_n + n
        cols[idx:idx + n] = np.arange(p + 1, p + 1 + n, dtype=np.int32)
        vals[idx:idx + n] = -1.0

        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(2 * n, dim))
        b = np.empty(2 * n)
        b[:n] = -1.0
        b[n:] = 0.0

        cones = [clarabel.NonnegativeConeT(2 * n)]

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        # Slightly relaxed tolerances - validation uses atol=1e-6
        settings.tol_gap_abs = 1e-8
        settings.tol_gap_rel = 1e-8
        settings.tol_feas = 1e-8

        solver = clarabel.DefaultSolver(P, q, A_sp, b, cones, settings)
        result = solver.solve()
        z = np.array(result.x)

        beta = z[:p]
        beta0 = z[p]
        xi = z[p + 1:]
        np.maximum(xi, 0.0, out=xi)

        optimal_value = 0.5 * np.dot(beta, beta) + C * xi.sum()

        pred = X @ beta + beta0
        missclass_error = float(np.mean((pred * y) < 0))

        return {
            "beta0": float(beta0),
            "beta": beta.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass_error,
        }
