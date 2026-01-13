"""
N-Queens Solver - Gen1a: Inlined MRV with Loop Unrolling

Mutation: Remove function call overhead by inlining count_available
and get_available directly into the backtrack loop. Also uses early
termination in MRV selection when finding rows with 1 available position.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using inlined MRV heuristic.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n

    # Track conflicts using arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)  # row - col + n - 1
    diag2_used = [False] * (2 * n - 1)  # row + col

    row_done = [False] * n
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

            # Inlined count_available + get_available
            available = []
            base_d1 = row + n_minus_1
            for col in range(n):
                if not col_used[col] and not diag1_used[base_d1 - col] and not diag2_used[row + col]:
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
