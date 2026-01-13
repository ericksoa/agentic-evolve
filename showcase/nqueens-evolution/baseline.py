"""
Baseline N-Queens Solver - Simple backtracking.

This is a basic recursive backtracking solution.
The goal is to evolve faster solvers that generalize across board sizes.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve the N-Queens problem.

    Args:
        n: Board size (n x n board with n queens)

    Returns:
        List of column positions for each row, or None if no solution.
        E.g., [1, 3, 0, 2] means queen in row 0 is at column 1, etc.
    """
    def is_safe(board: list[int], row: int, col: int) -> bool:
        """Check if placing a queen at (row, col) is safe."""
        for prev_row in range(row):
            prev_col = board[prev_row]
            # Same column
            if prev_col == col:
                return False
            # Diagonal
            if abs(prev_col - col) == abs(prev_row - row):
                return False
        return True

    def backtrack(board: list[int], row: int) -> bool:
        """Try to place queens starting from the given row."""
        if row == n:
            return True  # All queens placed

        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                if backtrack(board, row + 1):
                    return True
                board[row] = -1  # Backtrack

        return False

    board = [-1] * n
    if backtrack(board, 0):
        return board
    return None


def count_solutions(n: int, limit: int = 1000) -> int:
    """
    Count solutions to N-Queens (up to limit).

    Args:
        n: Board size
        limit: Stop counting after this many solutions

    Returns:
        Number of solutions found (capped at limit)
    """
    count = 0

    def is_safe(board: list[int], row: int, col: int) -> bool:
        for prev_row in range(row):
            prev_col = board[prev_row]
            if prev_col == col:
                return False
            if abs(prev_col - col) == abs(prev_row - row):
                return False
        return True

    def backtrack(board: list[int], row: int) -> None:
        nonlocal count
        if count >= limit:
            return

        if row == n:
            count += 1
            return

        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
                board[row] = -1

    board = [-1] * n
    backtrack(board, 0)
    return count


def verify_solution(n: int, solution: list[int]) -> bool:
    """Verify that a solution is valid."""
    if solution is None or len(solution) != n:
        return False

    for row, col in enumerate(solution):
        if col < 0 or col >= n:
            return False
        # Check conflicts with previous queens
        for prev_row in range(row):
            prev_col = solution[prev_row]
            if prev_col == col:
                return False
            if abs(prev_col - col) == abs(prev_row - row):
                return False

    return True


# For testing when run directly
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        # Evaluation mode
        import json

        sizes = [8, 12, 16, 20]
        results = {}
        total_time = 0

        for n in sizes:
            start = time.perf_counter()
            solution = solve_nqueens(n)
            elapsed = time.perf_counter() - start
            total_time += elapsed

            valid = verify_solution(n, solution)
            results[f"n{n}"] = {
                "valid": valid,
                "time_ms": elapsed * 1000,
                "solution": solution if valid else None,
            }

        # Fitness: solutions per second (higher is better)
        fitness = len(sizes) / total_time if total_time > 0 else 0

        print(json.dumps({
            "valid": all(r["valid"] for r in results.values()),
            "fitness": fitness,
            "total_time_ms": total_time * 1000,
            "results": results,
        }))
    else:
        # Demo mode
        for n in [8, 12, 16]:
            start = time.perf_counter()
            solution = solve_nqueens(n)
            elapsed = time.perf_counter() - start
            print(f"N={n}: {solution} ({elapsed*1000:.2f}ms)")
