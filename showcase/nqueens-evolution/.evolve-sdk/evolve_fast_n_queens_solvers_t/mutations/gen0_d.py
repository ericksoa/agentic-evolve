"""
N-Queens Solver - Variant D: Permutation-based with Early Pruning

Approach: Treats solution as permutation of columns.
- Each row gets exactly one queen (permutation)
- Only checks diagonal conflicts during placement
- Uses swap-based candidate generation
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens by generating permutations with pruning.

    Since each row and column must have exactly one queen,
    the solution is a permutation of [0, 1, ..., n-1].
    We just need to ensure no diagonal conflicts.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = list(range(n))  # Start with identity permutation

    # Track diagonal conflicts
    diag1_used = [False] * (2 * n - 1)  # row - col + n - 1
    diag2_used = [False] * (2 * n - 1)  # row + col

    def backtrack(row: int) -> bool:
        if row == n:
            return True

        # Try swapping current position with each remaining position
        for i in range(row, n):
            col = solution[i]
            d1 = row - col + n - 1
            d2 = row + col

            if not diag1_used[d1] and not diag2_used[d2]:
                # Swap to put this column at current row
                solution[row], solution[i] = solution[i], solution[row]
                diag1_used[d1] = True
                diag2_used[d2] = True

                if backtrack(row + 1):
                    return True

                # Undo
                diag1_used[d1] = False
                diag2_used[d2] = False
                solution[row], solution[i] = solution[i], solution[row]

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
