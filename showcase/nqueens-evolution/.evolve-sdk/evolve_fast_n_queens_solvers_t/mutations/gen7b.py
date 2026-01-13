"""
N-Queens Solver - Gen7b: Branch Elimination with Integer Arithmetic

Parent: gen1a (fitness 18722.023755)
Mutation: Replace boolean arrays with integer arrays (0/1) and use
arithmetic multiplication instead of `and not` conditions.

Instead of: if not col_used[col] and not diag1_used[d1] and not diag2_used[d2]
We use:     if (1 - col_used[col]) * (1 - diag1_used[d1]) * (1 - diag2_used[d2])

This eliminates short-circuit evaluation branches and can be faster when
the CPU branch predictor struggles with the pattern of queen placements.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using branch-reduced MRV heuristic.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n

    # Use integer arrays (0=free, 1=used) instead of boolean
    col_used = [0] * n
    diag1_used = [0] * (2 * n - 1)  # row - col + n - 1
    diag2_used = [0] * (2 * n - 1)  # row + col

    row_done = [0] * n
    n_minus_1 = n - 1  # Precompute

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

            # Branch-eliminated availability check using multiplication
            available = []
            base_d1 = row + n_minus_1
            for col in range(n):
                d1 = base_d1 - col
                d2 = row + col
                # Multiply: result is 1 only if all are 0 (free)
                # This avoids short-circuit evaluation branches
                if (1 - col_used[col]) * (1 - diag1_used[d1]) * (1 - diag2_used[d2]):
                    available.append(col)

            count = len(available)
            if count == 0:
                # Row with no options - immediate fail indicator
                min_count = 0
                min_row = row
                min_available = available
                break
            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                if count == 1:
                    # Can't do much better than 1 in practice
                    break

        if min_row == -1 or min_count == 0:
            return False

        row = min_row
        row_done[row] = 1
        base_d1 = row + n_minus_1

        for col in min_available:
            d1 = base_d1 - col
            d2 = row + col

            solution[row] = col
            col_used[col] = 1
            diag1_used[d1] = 1
            diag2_used[d2] = 1

            if backtrack(placed + 1):
                return True

            col_used[col] = 0
            diag1_used[d1] = 0
            diag2_used[d2] = 0

        solution[row] = -1
        row_done[row] = 0
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
