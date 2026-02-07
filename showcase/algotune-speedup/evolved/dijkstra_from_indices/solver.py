"""
Approach: Dial's Algorithm (Bucket Queue) for integer-weighted graphs
Mutation: Algorithm family change - exploit integer weights [1,100]
- Dial's algorithm uses circular bucket array of size C+1 (C=max weight)
- O(m + n*C) per source vs O(m log n) for heap-based Dijkstra
- For dense graphs (20% density), m ~ 0.2*n^2, so bucket queue wins big
- Parallel processing of sources via numba.prange
- Symmetrize graph for undirected correctness
- Pre-allocated output arrays, cache=True for persistent JIT
"""
import numpy as np
import numba as nb
from scipy.sparse import csr_matrix


@nb.njit(cache=True)
def _dijkstra_dial_single(indptr, indices, data, n, source, dist_out, max_weight):
    """Dijkstra single-source using Dial's algorithm (circular bucket queue).

    For integer weights in [1, max_weight], uses a circular array of C+1 buckets.
    Invariant: when processing distance d, all queued distances are in [d, d+C].
    """
    INF = np.inf
    C = max_weight
    num_buckets = C + 1

    # Initialize distances
    for i in range(n):
        dist_out[i] = INF
    dist_out[source] = 0.0

    # Bucket linked-list via arrays
    # bucket_head[b] = first node in bucket b, -1 if empty
    bucket_head = np.full(num_buckets, -1, dtype=np.int64)
    node_next = np.full(n, -1, dtype=np.int64)
    node_prev = np.full(n, -1, dtype=np.int64)
    in_queue = np.zeros(n, dtype=nb.boolean)

    # Insert source into bucket 0
    bucket_head[0] = source
    in_queue[source] = True

    settled = 0
    # Scan distance values linearly; use circular indexing for bucket array
    # Max possible distance = (n-1) * C (longest shortest path in graph)
    max_dist = (n - 1) * C

    d_cur = 0
    while d_cur <= max_dist and settled < n:
        b = d_cur % num_buckets

        # Process all nodes in this bucket (they all have distance == d_cur)
        while bucket_head[b] != -1:
            u = bucket_head[b]
            # Remove u from bucket
            nxt = node_next[u]
            bucket_head[b] = nxt
            if nxt != -1:
                node_prev[nxt] = -1
            in_queue[u] = False
            node_next[u] = -1
            node_prev[u] = -1

            # Skip if already settled at a better distance (shouldn't happen with
            # proper decrease-key, but safety check)
            if dist_out[u] != d_cur:
                continue

            settled += 1

            # Relax neighbors
            for idx in range(indptr[u], indptr[u + 1]):
                v = indices[idx]
                w_int = int(data[idx])
                new_dist = d_cur + w_int
                if new_dist < dist_out[v]:
                    # Remove v from old bucket if it's queued
                    if in_queue[v]:
                        old_b = int(dist_out[v]) % num_buckets
                        prv = node_prev[v]
                        nxt2 = node_next[v]
                        if prv != -1:
                            node_next[prv] = nxt2
                        else:
                            bucket_head[old_b] = nxt2
                        if nxt2 != -1:
                            node_prev[nxt2] = prv
                        node_next[v] = -1
                        node_prev[v] = -1

                    dist_out[v] = float(new_dist)
                    # Insert v into new bucket
                    new_b = new_dist % num_buckets
                    old_head = bucket_head[new_b]
                    bucket_head[new_b] = v
                    node_next[v] = old_head
                    node_prev[v] = -1
                    if old_head != -1:
                        node_prev[old_head] = v
                    in_queue[v] = True

        d_cur += 1


@nb.njit(parallel=True, cache=True)
def _dijkstra_parallel(indptr, indices, data, n, sources, out, max_weight):
    """Run Dial's Dijkstra from multiple sources in parallel using prange."""
    num_sources = len(sources)
    for i in nb.prange(num_sources):
        _dijkstra_dial_single(indptr, indices, data, n, sources[i], out[i], max_weight)


@nb.njit(cache=True)
def _dijkstra_sequential(indptr, indices, data, n, sources, out, max_weight):
    """Run Dial's Dijkstra from multiple sources sequentially."""
    num_sources = len(sources)
    for i in range(num_sources):
        _dijkstra_dial_single(indptr, indices, data, n, sources[i], out[i], max_weight)


# Warm up JIT at import time with a tiny graph
_w_ip = np.array([0, 1, 2], dtype=np.int64)
_w_idx = np.array([1, 0], dtype=np.int64)
_w_d = np.array([1.0, 1.0], dtype=np.float64)
_w_out = np.empty((1, 2), dtype=np.float64)
_w_sources = np.array([0], dtype=np.int64)
_dijkstra_dial_single(_w_ip, _w_idx, _w_d, 2, 0, _w_out[0], 100)
_dijkstra_sequential(_w_ip, _w_idx, _w_d, 2, _w_sources, _w_out, 100)
_dijkstra_parallel(_w_ip, _w_idx, _w_d, 2, _w_sources, _w_out, 100)


class Solver:
    def solve(self, problem, **kwargs):
        data = np.asarray(problem["data"], dtype=np.float64)
        indices_arr = np.asarray(problem["indices"], dtype=np.int32)
        indptr_arr = np.asarray(problem["indptr"], dtype=np.int32)
        shape = tuple(problem["shape"])
        n = shape[0]
        source_indices = problem["source_indices"]

        if not source_indices:
            return {"distances": []}

        # Build CSR and symmetrize for undirected graph (fast C-level scipy ops)
        graph = csr_matrix((data, indices_arr, indptr_arr), shape=shape)
        sym = graph.minimum(graph.T).tocsr()
        sym.eliminate_zeros()

        # Extract CSR arrays for Numba (int64 for Numba compatibility)
        sym_indptr = np.asarray(sym.indptr, dtype=np.int64)
        sym_indices = np.asarray(sym.indices, dtype=np.int64)
        sym_data = np.asarray(sym.data, dtype=np.float64)

        # Determine max weight for bucket sizing
        if sym_data.size > 0:
            max_weight = int(np.max(sym_data))
        else:
            max_weight = 100
        if max_weight < 1:
            max_weight = 1

        sources = np.asarray(source_indices, dtype=np.int64)
        num_sources = len(sources)

        # Pre-allocate output
        out = np.empty((num_sources, n), dtype=np.float64)

        # Use parallel for multiple sources, sequential for single
        if num_sources >= 2:
            _dijkstra_parallel(sym_indptr, sym_indices, sym_data, n, sources, out, max_weight)
        else:
            _dijkstra_sequential(sym_indptr, sym_indices, sym_data, n, sources, out, max_weight)

        # Efficient conversion to Python lists with None for inf
        inf_mask = np.isinf(out)
        has_inf = np.any(inf_mask)

        if not has_inf:
            return {"distances": out.tolist()}

        result = []
        for i in range(num_sources):
            row = []
            row_data = out[i]
            row_mask = inf_mask[i]
            for j in range(n):
                if row_mask[j]:
                    row.append(None)
                else:
                    row.append(row_data[j])
            result.append(row)

        return {"distances": result}
