"""
Mutation gen1a: Fix sorting + streamline monotone chain
- Parent: gen0_a.py (fitness=0, likely broken by combined float sort key)
- Mutation: Replace fragile _make_sort_key with np.lexsort (correct lexicographic ordering)
- Also: inline cross product, avoid redundant copies, streamline output path
- Uses Numba JIT for the core monotone chain algorithm
"""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def _monotone_chain(px, py, n, order):
    """
    Andrew's monotone chain algorithm.
    px, py: x and y coordinates (float64 arrays)
    order: indices that sort points by (x, y) lexicographically
    Returns hull indices (counter-clockwise) and count.
    """
    hull = np.empty(2 * n, dtype=nb.int64)
    k = 0

    # Build lower hull
    for i in range(n):
        idx = order[i]
        while k >= 2:
            # Inline cross product: (hull[k-1] - hull[k-2]) x (idx - hull[k-2])
            ox = px[hull[k - 2]]
            oy = py[hull[k - 2]]
            cross = (px[hull[k - 1]] - ox) * (py[idx] - oy) - (py[hull[k - 1]] - oy) * (px[idx] - ox)
            if cross <= 0:
                k -= 1
            else:
                break
        hull[k] = idx
        k += 1

    # Build upper hull
    lower_size = k + 1
    for i in range(n - 2, -1, -1):
        idx = order[i]
        while k >= lower_size:
            ox = px[hull[k - 2]]
            oy = py[hull[k - 2]]
            cross = (px[hull[k - 1]] - ox) * (py[idx] - oy) - (py[hull[k - 1]] - oy) * (px[idx] - ox)
            if cross <= 0:
                k -= 1
            else:
                break
        hull[k] = idx
        k += 1

    return hull, k - 1


@nb.njit(cache=True)
def _ensure_ccw(hull_indices, count, px, py):
    """Ensure counter-clockwise ordering by checking signed area."""
    area = 0.0
    for i in range(count):
        j = (i + 1) % count
        hi = hull_indices[i]
        hj = hull_indices[j]
        area += px[hi] * py[hj]
        area -= px[hj] * py[hi]

    if area < 0:
        # Reverse in place
        for i in range(count // 2):
            j = count - 1 - i
            tmp = hull_indices[i]
            hull_indices[i] = hull_indices[j]
            hull_indices[j] = tmp

    return hull_indices


# Pre-warm JIT with a small problem
_dummy_x = np.array([0.0, 1.0, 0.5, 0.3], dtype=np.float64)
_dummy_y = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float64)
_dummy_order = np.lexsort((_dummy_y, _dummy_x)).astype(np.int64)
_dummy_hull, _dummy_count = _monotone_chain(_dummy_x, _dummy_y, 4, _dummy_order)
_ensure_ccw(_dummy_hull[:_dummy_count], _dummy_count, _dummy_x, _dummy_y)


class Solver:
    def solve(self, problem):
        points = np.asarray(problem["points"], dtype=np.float64)
        n = len(points)

        if n < 3:
            hull_vertices = list(range(n))
            hull_points = points.tolist()
            return {"hull_vertices": hull_vertices, "hull_points": hull_points}

        px = np.ascontiguousarray(points[:, 0])
        py = np.ascontiguousarray(points[:, 1])

        # Use np.lexsort for correct lexicographic (x, y) ordering
        # lexsort sorts by last key first, so (py, px) gives sort-by-x-then-y
        order = np.lexsort((py, px)).astype(np.int64)

        hull, count = _monotone_chain(px, py, n, order)

        hv = hull[:count].copy()
        hv = _ensure_ccw(hv, count, px, py)

        hull_vertices = hv.tolist()
        hull_points = points[hv].tolist()

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}
