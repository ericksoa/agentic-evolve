"""N-Queens solver - BROKEN mutation (type error)."""


def solve_n_queens(n: int) -> list[list[int]]:
    """Find all solutions to N-Queens problem."""
    solutions = []

    def is_safe(board: list[int], row: int, col: int) -> bool:
        for prev_row in range(row):
            prev_col = board[prev_row]
            if prev_col == col:
                return False
            if abs(prev_col - col) == abs(prev_row - row):
                return False
        return True

    def backtrack(board: list[int], row: int):
        if row == n:
            solutions.append(board.copy())
            return

        # BUG: Tried to "optimize" by using set, but range() returns ints
        # and we're iterating incorrectly
        for col in set(range(n)):  # set() is fine but...
            if is_safe(board, row, col):
                board.append(col)
                backtrack(board, row + "1")  # <-- TYPE ERROR: can't add str to int
                board.pop()

    backtrack([], 0)
    return solutions


if __name__ == "__main__":
    solutions = solve_n_queens(8)
    print(f"Found {len(solutions)} solutions")
