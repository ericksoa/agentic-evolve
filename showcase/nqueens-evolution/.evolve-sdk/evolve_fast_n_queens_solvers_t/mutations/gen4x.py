"""
N-Queens Solver - Gen4x: Optimized Hybrid Crossover

Combines best aspects from three parents:
- gen1a: Inlined MRV with boolean arrays (fastest for large n)
- gen3x: Hybrid dispatch concept
- gen2b: Efficient bitwise backtracking for small n

Key innovations in this crossover:
1. Boolean arrays for conflict tracking (gen1a) - faster than bitmasks for large n
2. Fully inlined MRV selection with early termination (gen1a)
3. Optimized threshold (n<=8) for bitwise vs MRV dispatch
4. Precomputed diagonal offsets (n_minus_1) from gen1a
5. Immediate fail detection when any row has 0 available positions
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using optimized hybrid approach.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # From gen2b/gen3x: Use bitwise for small n (lower overhead)
    # Optimized threshold: 8 works better than 10 in practice
    if n <= 8:
        return _solve_bitwise(n)
    else:
        return _solve_mrv_inlined(n)


def _solve_bitwise(n: int) -> list[int] | None:
    """From gen2b: Simple row-by-row backtracking with bitmask conflict tracking."""
    solution = [-1] * n
    all_cols = (1 << n) - 1

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            bit = available & -available
            available ^= bit
            col = (bit - 1).bit_count()
            solution[row] = col

            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_inlined(n: int) -> list[int] | None:
    """
    From gen1a: MRV solver with fully inlined heuristic and boolean arrays.
    This is the fastest approach for larger boards.
    """
    solution = [-1] * n

    # From gen1a: Boolean arrays (faster than bitmasks for large n)
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n
    n_minus_1 = n - 1  # Precompute for diagonal calculation

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # From gen1a: Fully inlined MRV selection
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            # Inlined availability check (from gen1a)
            available = []
            base_d1 = row + n_minus_1
            for col in range(n):
                if not col_used[col] and not diag1_used[base_d1 - col] and not diag2_used[row + col]:
                    available.append(col)

            count = len(available)

            # From gen1a: Immediate fail when row has no options
            if count == 0:
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                # From gen1a: Early termination - can't do better than 1
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
