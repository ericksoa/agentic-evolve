"""
N-Queens Solver - Gen10c: Lookup Table for Bit Position

Parent: gen8b (fitness 18755.156)

Mutation type: Lookup table optimization
Changed: Replace bit.bit_length() - 1 with a precomputed lookup table

Hypothesis: The bit.bit_length() call in the inner loop has overhead from
Python method dispatch. Using a direct dictionary lookup for bit -> column
mapping eliminates this overhead. For small n values that use the bitwise
solver, the lookup table is small and stays in L1 cache.

Additionally, cache the all_cols value and use a local reference for
faster variable access in the inner loop.
"""

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
    """Simple row-by-row backtracking with bitmask conflict tracking.

    Optimization: Precompute bit -> column lookup table to avoid
    bit_length() calls in the hot path.
    """
    solution = [-1] * n
    all_cols = (1 << n) - 1

    # Precompute bit position lookup table: bit -> column index
    # For n <= 12, this is at most 12 entries
    bit_to_col = {1 << i: i for i in range(n)}

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
