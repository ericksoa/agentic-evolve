"""
N-Queens Solver - Gen8a: Eliminate List Construction in MRV

Mutation from gen6x: Remove available[] list construction overhead in MRV loop.

Key change: Instead of building available[] list during MRV row selection,
just count available columns. Then re-scan during placement phase.
This eliminates list append operations in the hot inner loop.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach combining best of all parents.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # From gen5b: Use simple bitwise for small n (lower overhead)
    if n <= 10:
        return _solve_simple_bitwise(n)
    else:
        return _solve_mrv_precomputed(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking.

    From gen5b: Uses bit.bit_length() - 1 for faster bit position calculation.
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

            # From gen5b: bit_length() - 1 is faster than (bit-1).bit_count()
            col = bit.bit_length() - 1
            solution[row] = col

            # Recurse: shift diagonals for next row
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_precomputed(n: int) -> list[int] | None:
    """
    Inlined MRV with precomputed diagonal lookup tables.

    Gen8a mutation: Eliminate list construction in MRV inner loop.
    Instead of building available[] list, just count and re-scan during placement.
    """
    solution = [-1] * n

    # From gen5a: Precompute diagonal lookup tables to eliminate arithmetic
    n_minus_1 = n - 1
    d1_table = [[row - col + n_minus_1 for col in range(n)] for row in range(n)]
    d2_table = [[row + col for col in range(n)] for row in range(n)]

    # Track conflicts using boolean arrays (from gen1a - simpler than bitmasks for large n)
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # MRV selection: find row with minimum available columns
        # Gen8a: Only count, don't build list (eliminates append overhead)
        min_count = n + 1
        min_row = -1

        for row in range(n):
            if row_done[row]:
                continue

            d1_row = d1_table[row]
            d2_row = d2_table[row]
            count = 0

            # Just count available columns, don't build list
            for col in range(n):
                if not col_used[col] and not diag1_used[d1_row[col]] and not diag2_used[d2_row[col]]:
                    count += 1

            # Immediate fail when row has no options
            if count == 0:
                return False

            if count < min_count:
                min_count = count
                min_row = row
                # Early termination when count == 1
                if count == 1:
                    break

        if min_row == -1:
            return False

        row = min_row
        row_done[row] = True
        d1_row = d1_table[row]
        d2_row = d2_table[row]

        # Gen8a: Re-scan to find available columns for placement
        # Trade-off: extra scan vs list construction overhead
        for col in range(n):
            if col_used[col] or diag1_used[d1_row[col]] or diag2_used[d2_row[col]]:
                continue

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
