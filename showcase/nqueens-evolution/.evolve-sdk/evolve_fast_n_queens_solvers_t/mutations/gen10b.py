"""
N-Queens Solver - Gen10b: Inline Diagonal Computation

Parent: gen9a (fitness 20271.558)

Mutation type: Cache optimization
Changed: Replace 2D diagonal lookup tables with inline computation

Hypothesis: The parent uses d1_table[row][col] and d2_table[row][col] which
require two pointer dereferences per access. This can hurt cache locality
especially since we access different rows each iteration.

Instead, compute diagonals inline:
- d1 = row - col + (n - 1)  (simple arithmetic)
- d2 = row + col            (single addition)

Modern CPUs can do these calculations faster than cache misses from
table lookups, especially when the table spans multiple cache lines.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach with inline diagonal computation.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Use simple bitwise for small n (lower overhead)
    if n <= 10:
        return _solve_simple_bitwise(n)
    else:
        return _solve_mrv_inline(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking."""
    solution = [-1] * n
    all_cols = (1 << n) - 1

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            bit = available & -available
            available ^= bit
            col = bit.bit_length() - 1
            solution[row] = col

            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_inline(n: int) -> list[int] | None:
    """
    MRV with inline diagonal computation instead of lookup tables.

    Key change: Diagonals computed as:
    - d1 = row - col + n_minus_1  (anti-diagonal)
    - d2 = row + col              (main diagonal)

    This avoids 2D array lookups and improves cache locality.
    """
    solution = [-1] * n
    n_minus_1 = n - 1

    # Track conflicts using boolean arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n

    # Precompute n mod 4 for unrolling remainder
    n_unroll = n & ~3  # Largest multiple of 4 <= n

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # Inlined MRV selection with loop unrolling
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            # Inline diagonal base computation
            base_d1 = row + n_minus_1
            available = []

            # Unrolled loop: process 4 columns at a time
            col = 0
            while col < n_unroll:
                # Check col
                if not col_used[col] and not diag1_used[base_d1 - col] and not diag2_used[row + col]:
                    available.append(col)
                # Check col+1
                c1 = col + 1
                if not col_used[c1] and not diag1_used[base_d1 - c1] and not diag2_used[row + c1]:
                    available.append(c1)
                # Check col+2
                c2 = col + 2
                if not col_used[c2] and not diag1_used[base_d1 - c2] and not diag2_used[row + c2]:
                    available.append(c2)
                # Check col+3
                c3 = col + 3
                if not col_used[c3] and not diag1_used[base_d1 - c3] and not diag2_used[row + c3]:
                    available.append(c3)
                col += 4

            # Handle remainder columns
            while col < n:
                if not col_used[col] and not diag1_used[base_d1 - col] and not diag2_used[row + col]:
                    available.append(col)
                col += 1

            count = len(available)

            # Immediate fail when row has no options
            if count == 0:
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                # Early termination when count == 1
                if count == 1:
                    break

        if min_row == -1:
            return False

        row = min_row
        row_done[row] = True
        base_d1 = row + n_minus_1

        for col in min_available:
            d1 = base_d1 - col
            d2 = row + col

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
