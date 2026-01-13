"""
N-Queens Solver - Gen7a: Loop Unrolling Mutation

Parent: gen6x (fitness 20406.72843)
Mutation: Loop unrolling in MRV column availability check

The inner loop checking column availability is a hot path. By unrolling it
4x, we reduce loop overhead and can batch memory accesses more efficiently.
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

    Gen7a mutation: Loop unrolling in the column availability check.
    Process 4 columns at a time to reduce loop overhead.
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

    # Precompute loop unrolling boundaries
    n_unrolled = (n >> 2) << 2  # n rounded down to multiple of 4

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

            # From gen5a: Use precomputed lookup tables instead of computing indices
            d1_row = d1_table[row]
            d2_row = d2_table[row]
            available = []

            # Gen7a: Unrolled loop - process 4 columns at a time
            col = 0
            while col < n_unrolled:
                # Unroll 4 iterations
                d1_0 = d1_row[col]
                d2_0 = d2_row[col]
                if not col_used[col] and not diag1_used[d1_0] and not diag2_used[d2_0]:
                    available.append(col)

                col1 = col + 1
                d1_1 = d1_row[col1]
                d2_1 = d2_row[col1]
                if not col_used[col1] and not diag1_used[d1_1] and not diag2_used[d2_1]:
                    available.append(col1)

                col2 = col + 2
                d1_2 = d1_row[col2]
                d2_2 = d2_row[col2]
                if not col_used[col2] and not diag1_used[d1_2] and not diag2_used[d2_2]:
                    available.append(col2)

                col3 = col + 3
                d1_3 = d1_row[col3]
                d2_3 = d2_row[col3]
                if not col_used[col3] and not diag1_used[d1_3] and not diag2_used[d2_3]:
                    available.append(col3)

                col += 4

            # Handle remaining columns (if n is not divisible by 4)
            while col < n:
                d1_c = d1_row[col]
                d2_c = d2_row[col]
                if not col_used[col] and not diag1_used[d1_c] and not diag2_used[d2_c]:
                    available.append(col)
                col += 1

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
