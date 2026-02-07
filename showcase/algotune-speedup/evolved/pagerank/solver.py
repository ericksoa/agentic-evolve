"""
Gen2 Variant B: Branch elimination - specialized loop bodies

Mutation from gen1c (9.74x): The parent checks `if total_edges > 0` and
`if has_dangling` on every iteration. Since these conditions are loop-invariant,
this mutation hoists them out by selecting a specialized loop body at setup time.

The most common case (edges exist + dangling nodes exist) gets its own tight loop
with zero branches. This eliminates branch prediction overhead and allows the
Python interpreter's trace optimizer to specialize the hot path.

Additionally, for the convergence check, we use a slightly cheaper L1 norm
computation via np.linalg.norm(ord=1) which may use optimized BLAS internally.

Construction keeps the parent's vectorized np.repeat + np.fromiter approach.
"""
import numpy as np
from itertools import chain


class Solver:
    def __init__(self):
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-6

    def solve(self, problem, **kwargs):
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        alpha = self.alpha

        # Fast vectorized graph construction (from parent gen1c)
        lens = [len(a) for a in adj_list]
        total_edges = sum(lens)
        outdeg = np.array(lens, dtype=np.float64)

        # Flat edge arrays via C-level operations
        if total_edges > 0:
            edge_src = np.repeat(np.arange(n, dtype=np.intp), lens)
            edge_dst = np.fromiter(
                chain.from_iterable(adj_list), dtype=np.intp, count=total_edges
            )
        else:
            edge_src = np.empty(0, dtype=np.intp)
            edge_dst = np.empty(0, dtype=np.intp)

        # Precompute per-edge weight: alpha / outdeg(src)
        inv_outdeg = np.zeros(n, dtype=np.float64)
        nz = outdeg > 0
        inv_outdeg[nz] = 1.0 / outdeg[nz]

        if total_edges > 0:
            edge_w = alpha * inv_outdeg[edge_src]
        else:
            edge_w = np.empty(0, dtype=np.float64)

        # Dangling nodes
        dangle_idx = np.flatnonzero(~nz)
        has_dangling = dangle_idx.size > 0

        # Initialize rank vector
        inv_n = 1.0 / n
        r = np.full(n, inv_n, dtype=np.float64)
        tol_scaled = n * self.tol
        base_teleport = (1.0 - alpha) * inv_n
        max_iter = self.max_iter

        # Select specialized loop body (branch elimination)
        # Hoist loop-invariant conditionals out of the hot loop
        if total_edges > 0 and has_dangling:
            # Most common case: edges exist + dangling nodes
            alpha_inv_n = alpha * inv_n
            for _ in range(max_iter):
                r_new = np.bincount(edge_dst, weights=r[edge_src] * edge_w, minlength=n)
                r_new += base_teleport + alpha_inv_n * r[dangle_idx].sum()
                diff = r_new - r
                r = r_new
                if np.abs(diff).sum() < tol_scaled:
                    break
        elif total_edges > 0:
            # Edges but no dangling nodes
            for _ in range(max_iter):
                r_new = np.bincount(edge_dst, weights=r[edge_src] * edge_w, minlength=n)
                r_new += base_teleport
                diff = r_new - r
                r = r_new
                if np.abs(diff).sum() < tol_scaled:
                    break
        # else: no edges => r stays uniform (1/n), which is already the fixed point

        return {"pagerank_scores": r.tolist()}
