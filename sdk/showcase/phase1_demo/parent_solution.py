"""N-Queens solver - working parent solution."""


def solve_n_queens(n: int) -> list[list[int]]:
    """Find all solutions to N-Queens problem.

    Returns list of solutions, each solution is a list of column positions.
    """
    solutions = []

    def is_safe(board: list[int], row: int, col: int) -> bool:
        for prev_row in range(row):
            prev_col = board[prev_row]
            # Same column
            if prev_col == col:
                return False
            # Diagonal
            if abs(prev_col - col) == abs(prev_row - row):
                return False
        return True

    def backtrack(board: list[int], row: int):
        if row == n:
            solutions.append(board.copy())
            return

        for col in range(n):
            if is_safe(board, row, col):
                board.append(col)
                backtrack(board, row + 1)
                board.pop()

    backtrack([], 0)
    return solutions


if __name__ == "__main__":
    import time
    start = time.perf_counter()
    solutions = solve_n_queens(10)
    elapsed = time.perf_counter() - start
    print(f"Found {len(solutions)} solutions in {elapsed:.4f}s")
