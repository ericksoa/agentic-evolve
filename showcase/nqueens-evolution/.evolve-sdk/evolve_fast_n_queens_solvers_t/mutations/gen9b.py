"""
N-Queens Solver - Gen9b: Lookup Table for Bit-to-Column Conversion

Parent: gen8b (fitness 18755.16)
Mutation: Replace bit.bit_length()-1 with a precomputed lookup table.

Key insight: The bit_length() method is a function call that adds overhead.
For n <= 20 (typical board sizes), we can precompute a lookup table that
maps powers of 2 to their log2 values. This eliminates the function call
in the hot path of the bitwise solver.

Uses (bit & -bit) which gives powers of 2: 1, 2, 4, 8, 16, ...
We can use bit.bit_length()-1 = log2(bit) for powers of 2.
Lookup table: 1->0, 2->1, 4->2, 8->3, etc.
"""

# Precompute bit-to-column lookup for common board sizes (up to n=30)
# Maps power of 2 -> log2(power of 2)
_BIT_TO_COL = {1 << i: i for i in range(30)}

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid dispatch.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Hybrid dispatch: bitwise for small n, MRV for large n
    if n <= 12:
        return _solve_bitwise(n)
    else:
        return _solve_mrv(n)


def _solve_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking."""
    solution = [-1] * n
    all_cols = (1 << n) - 1
    bit_to_col = _BIT_TO_COL  # Local reference for faster lookup

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            bit = available & -available
            available ^= bit
            col = bit_to_col[bit]
            solution[row] = col

            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv(n: int) -> list[int] | None:
    """
    MRV with inlined selection from parent gen1a.
    """
    solution = [-1] * n

    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n
    n_minus_1 = n - 1

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # Inlined MRV: select most constrained row
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            available = []
            base_d1 = row + n_minus_1
            for col in range(n):
                if not col_used[col] and not diag1_used[base_d1 - col] and not diag2_used[row + col]:
                    available.append(col)

            count = len(available)
            if count == 0:
                return False
            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
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
