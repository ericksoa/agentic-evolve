"""
Variant A: Bitwise Backtracking with Column/Diagonal Masks
Uses bitwise operations to track attacked columns and diagonals efficiently.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using bitwise backtracking."""
    board = [-1] * n
    all_cols = (1 << n) - 1  # All columns available

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        # Available positions: not in cols, diag1, or diag2
        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            # Get rightmost available position
            pos = available & -available
            available ^= pos

            # Convert to column index
            col = pos.bit_length() - 1
            board[row] = col

            # Recurse with updated masks
            if backtrack(row + 1,
                        cols | pos,
                        (diag1 | pos) << 1,
                        (diag2 | pos) >> 1):
                return True

            board[row] = -1

        return False

    if backtrack(0, 0, 0, 0):
        return board
    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
