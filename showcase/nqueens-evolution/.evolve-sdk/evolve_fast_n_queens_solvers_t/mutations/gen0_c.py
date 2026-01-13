"""
N-Queens Solver - Variant C: Array-based with MRV Heuristic

Approach: Uses Most Restricted Variable (MRV) heuristic.
- Prioritizes rows with fewer available positions
- Uses arrays instead of sets for faster access
- Precomputes conflict masks per column
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using MRV heuristic ordering.

    Instead of processing rows in order, pick the row
    with fewest available columns first.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n

    # Track conflicts using arrays (faster than sets for small n)
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)  # row - col + n - 1
    diag2_used = [False] * (2 * n - 1)  # row + col

    row_done = [False] * n

    def count_available(row: int) -> int:
        """Count available positions for a row."""
        count = 0
        for col in range(n):
            if not col_used[col] and not diag1_used[row - col + n - 1] and not diag2_used[row + col]:
                count += 1
        return count

    def get_available(row: int) -> list[int]:
        """Get list of available columns for a row."""
        result = []
        for col in range(n):
            if not col_used[col] and not diag1_used[row - col + n - 1] and not diag2_used[row + col]:
                result.append(col)
        return result

    def select_row() -> int:
        """Select row with minimum remaining values (MRV)."""
        min_count = n + 1
        min_row = -1
        for row in range(n):
            if not row_done[row]:
                count = count_available(row)
                if count < min_count:
                    min_count = count
                    min_row = row
                    if count == 0:
                        break  # Can't do better than 0
        return min_row

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # MRV: select most constrained row
        row = select_row()
        if row == -1:
            return False

        available = get_available(row)
        if not available:
            return False

        row_done[row] = True

        for col in available:
            d1 = row - col + n - 1
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
