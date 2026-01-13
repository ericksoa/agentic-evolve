"""
N-Queens Solver - Gen7x: Enhanced Crossover Hybrid

Combines the best aspects of three high-performing parents:
- gen6x (fitness 20406): Hybrid dispatch (bitwise for small n, MRV for large n)
- gen1a (fitness 18722): Inlined MRV with early termination optimizations
- gen5a (fitness 18120): Precomputed diagonal lookup tables

New optimizations in this generation:
1. Tuned dispatch threshold (n<=12 for bitwise, works better for more cases)
2. Flattened diagonal lookup tables (1D array instead of 2D for cache efficiency)
3. Local variable caching in hot loops to minimize global lookups
4. Combined the early break optimizations from gen1a with gen6x structure
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using enhanced hybrid approach.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Tuned threshold: bitwise works well up to n=12
    if n <= 12:
        return _solve_simple_bitwise(n)
    else:
        return _solve_mrv_optimized(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking.

    From gen6x: Uses bit.bit_length() - 1 for faster bit position calculation.
    """
    solution = [-1] * n
    all_cols = (1 << n) - 1  # Mask of all columns

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        # Available positions: columns not attacked
        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            # Get rightmost set bit (a valid column)
            bit = available & -available
            available ^= bit  # Remove this bit

            # From gen6x/gen5b: bit_length() - 1 is faster than (bit-1).bit_count()
            col = bit.bit_length() - 1
            solution[row] = col

            # Recurse: shift diagonals for next row
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_optimized(n: int) -> list[int] | None:
    """
    Optimized MRV with flattened diagonal lookup tables.

    Combines:
    - gen6x/gen5a: Precomputed diagonal lookup (now flattened for cache efficiency)
    - gen1a: Inlined MRV structure with early termination optimizations
    """
    solution = [-1] * n

    # Flattened diagonal lookup tables for better cache locality
    # Access: d1_flat[row * n + col] instead of d1_table[row][col]
    n_minus_1 = n - 1
    d1_flat = tuple(row - col + n_minus_1 for row in range(n) for col in range(n))
    d2_flat = tuple(row + col for row in range(n) for col in range(n))

    # Track conflicts using arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # Cache local references for faster access in inner loops
        _col_used = col_used
        _diag1_used = diag1_used
        _diag2_used = diag2_used
        _row_done = row_done
        _d1_flat = d1_flat
        _d2_flat = d2_flat
        _n = n

        # From gen1a: Inlined MRV selection with early termination
        min_count = _n + 1
        min_row = -1
        min_available = None

        for row in range(_n):
            if _row_done[row]:
                continue

            # Use flattened lookup tables
            base_idx = row * _n
            available = []

            for col in range(_n):
                idx = base_idx + col
                if not _col_used[col] and not _diag1_used[_d1_flat[idx]] and not _diag2_used[_d2_flat[idx]]:
                    available.append(col)

            count = len(available)

            # From gen1a: Immediate fail when row has no options
            if count == 0:
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                # From gen1a: Early termination when count == 1
                if count == 1:
                    break

        if min_row == -1:
            return False

        row = min_row
        _row_done[row] = True
        base_idx = row * _n

        for col in min_available:
            idx = base_idx + col
            d1 = _d1_flat[idx]
            d2 = _d2_flat[idx]

            solution[row] = col
            _col_used[col] = True
            _diag1_used[d1] = True
            _diag2_used[d2] = True

            if backtrack(placed + 1):
                return True

            _col_used[col] = False
            _diag1_used[d1] = False
            _diag2_used[d2] = False

        solution[row] = -1
        _row_done[row] = False
        return False

    if backtrack(0):
        return solution
    return None


def verify_solution(n: int, solution: list[int]) -> bool:
    """Verify that a solution is valid."""
    if solution is None or len(solution) != n:
        return False

    for row, col in enumerate(solution):
        if col < 0 or col >= n:
            return False
        for prev_row in range(row):
            prev_col = solution[prev_row]
            if prev_col == col:
                return False
            if abs(prev_col - col) == abs(prev_row - row):
                return False

    return True


if __name__ == "__main__":
    import time
    for n in [8, 12, 16, 20]:
        start = time.perf_counter()
        solution = solve_nqueens(n)
        elapsed = time.perf_counter() - start
        valid = verify_solution(n, solution)
        print(f"N={n}: valid={valid}, time={elapsed*1000:.2f}ms, solution={solution}")
