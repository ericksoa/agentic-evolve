"""N-Queens solver - BROKEN mutation (off-by-one error)."""


def solve_n_queens(n: int) -> list[list[int]]:
    """Find all solutions to N-Queens problem."""
    solutions = []

    def is_safe(board: list[int], row: int, col: int) -> bool:
        # BUG: range should be range(row), not range(row + 1)
        # This causes IndexError when accessing board[row] which doesn't exist yet
        for prev_row in range(row + 1):  # <-- OFF BY ONE ERROR
            prev_col = board[prev_row]   # <-- CRASHES HERE
            if prev_col == col:
                return False
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
    solutions = solve_n_queens(8)
    print(f"Found {len(solutions)} solutions")
