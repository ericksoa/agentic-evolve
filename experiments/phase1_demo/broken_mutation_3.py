"""N-Queens solver - BROKEN mutation (attribute error from bad refactor)."""


def solve_n_queens(n: int) -> list[list[int]]:
    """Find all solutions to N-Queens problem."""
    solutions = []

    # Attempted to use a class for "cleaner" code
    class Board:
        def __init__(self):
            self.positions = []

        def add(self, col):
            self.positions.append(col)

        def remove(self):
            self.positions.pop()

    def is_safe(board: Board, row: int, col: int) -> bool:
        for prev_row in range(row):
            prev_col = board.positions[prev_row]
            if prev_col == col:
                return False
            if abs(prev_col - col) == abs(prev_row - row):
                return False
        return True

    def backtrack(board: Board, row: int):
        if row == n:
            # BUG: .copy() doesn't exist on custom class, should be list(board.positions)
            solutions.append(board.positions.copy())  # This actually works
            return

        for col in range(n):
            if is_safe(board, row, col):
                board.add(col)
                backtrack(board, row + 1)
                board.removed()  # <-- ATTRIBUTE ERROR: typo, should be remove()

    board = Board()
    backtrack(board, 0)
    return solutions


if __name__ == "__main__":
    solutions = solve_n_queens(8)
    print(f"Found {len(solutions)} solutions")
