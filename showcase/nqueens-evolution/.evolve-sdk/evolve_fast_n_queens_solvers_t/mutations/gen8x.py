"""
N-Queens Solver - Gen8x: Crossover Hybrid

Combines the best aspects of three parents:
- gen6x (fitness 20406.72): Hybrid dispatch + bit_length() optimization
- gen1a (fitness 18722.02): Inlined MRV with early termination + base_d1 pattern
- gen5a (fitness 18120.44): Precomputed diagonal lookup tables

Crossover innovations:
1. Hybrid dispatch with tuned threshold (n<=12 for bitwise)
2. Bitwise solver with bit_length()-1 optimization (from gen6x)
3. MRV solver combines:
   - Precomputed lookup tables as tuples for faster access (gen5a)
   - Local variable caching in hot loops (all parents)
   - Early termination at count==0 immediately (gen1a)
   - Break at count==1 (gen1a/gen5a)
4. Direct list operations avoiding repeated lookups
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach combining best of all parents.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Tuned threshold: bitwise is efficient up to n=12
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

            # From gen6x: bit_length() - 1 is faster than (bit-1).bit_count()
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
    Optimized MRV combining all parent innovations.

    - gen5a: Precomputed diagonal tables (as tuples for speed)
    - gen1a: Early termination + break at count==1
    - All: Local variable caching in hot paths
    """
    solution = [-1] * n

    # From gen5a: Precomputed diagonal lookup tables as tuples (immutable, faster access)
    n_minus_1 = n - 1
    d1_table = tuple(tuple(row - col + n_minus_1 for col in range(n)) for row in range(n))
    d2_table = tuple(tuple(row + col for col in range(n)) for row in range(n))

    # Track conflicts using arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # From gen1a: Inlined MRV selection with early termination
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            # Cache row-specific lookups (from gen5a + local caching)
            d1_row = d1_table[row]
            d2_row = d2_table[row]
            available = []

            # Local variable caching for inner loop
            _col_used = col_used
            _diag1_used = diag1_used
            _diag2_used = diag2_used

            for col in range(n):
                if not _col_used[col] and not _diag1_used[d1_row[col]] and not _diag2_used[d2_row[col]]:
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
        row_done[row] = True
        d1_row = d1_table[row]
        d2_row = d2_table[row]

        for col in min_available:
            d1 = d1_row[col]
            d2 = d2_row[col]

            solution[row] = col
            col_used[col] = True
            diag1_used[d1] = True
            diag2_used[d2] = True

            if backtrack(placed + 1):
                return True

            col_used[col] = False
            diag1_used[d1] = False
            diag2_used[d2] = False

        solution[row] = -1
        row_done[row] = False
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
