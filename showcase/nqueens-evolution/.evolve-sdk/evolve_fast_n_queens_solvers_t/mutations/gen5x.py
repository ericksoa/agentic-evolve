"""
N-Queens Solver - Gen5x: Hybrid Crossover

Combines best aspects from three parents:
- gen2b/gen3x: Hybrid dispatch with simple bitwise backtracking for small n
- gen1a: Inlined MRV with boolean arrays (fastest for large n)
- gen1a: Precomputed n_minus_1 and early termination when count==1

Key insight: gen1a's boolean array approach outperforms bitmask operations
for the MRV solver. This hybrid uses bitwise for small n (low overhead)
and gen1a's optimized boolean-array MRV for larger n.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach with optimized dispatch.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # From gen2b/gen3x: Use simple bitwise backtracking for small n
    if n <= 12:
        return _solve_simple_bitwise(n)
    else:
        return _solve_mrv_inlined(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """From gen2b/gen3x: Simple row-by-row backtracking with bitmask conflict tracking."""
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

            # Convert bit position to column index
            col = (bit - 1).bit_count()
            solution[row] = col

            # Recurse: shift diagonals for next row
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_inlined(n: int) -> list[int] | None:
    """
    From gen1a: MRV solver with inlined heuristic using boolean arrays.
    Boolean arrays are faster than bitmask operations for the MRV pattern.
    """
    solution = [-1] * n

    # From gen1a: Track conflicts using boolean arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)  # row - col + n - 1
    diag2_used = [False] * (2 * n - 1)  # row + col

    row_done = [False] * n
    n_minus_1 = n - 1  # From gen1a: Precompute

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

            # From gen1a: Inlined count_available + get_available
            available = []
            base_d1 = row + n_minus_1
            for col in range(n):
                if not col_used[col] and not diag1_used[base_d1 - col] and not diag2_used[row + col]:
                    available.append(col)

            count = len(available)

            # From gen1a: Row with no options - immediate fail
            if count == 0:
                min_count = 0
                min_row = row
                min_available = available
                break

            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                # From gen1a: Early termination when count == 1
                if count == 1:
                    break

        if min_row == -1 or min_count == 0:
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
