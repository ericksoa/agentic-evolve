"""
Variant gen1b: Bitwise Backtracking (Algorithm Swap)
Replaces min-conflicts with pure bitwise backtracking for all sizes.
Bitwise operations eliminate array lookups and use fast bit manipulation.
"""


def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using bitwise backtracking."""
    if n <= 0:
        return None
    if n == 1:
        return [0]

    board = [-1] * n
    all_ones = (1 << n) - 1  # Mask with n bits set

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        """
        Bitwise backtracking solver.
        cols: columns already occupied (bit set = occupied)
        diag1: down-left diagonals occupied
        diag2: down-right diagonals occupied
        """
        if row == n:
            return True

        # Available positions: where no column or diagonal is blocked
        available = all_ones & ~(cols | diag1 | diag2)

        while available:
            # Get rightmost available position
            bit = available & (-available)
            available &= available - 1  # Remove this bit

            # Convert bit position to column index
            col = (bit - 1).bit_count()
            board[row] = col

            # Recurse with updated constraints
            # diag1 shifts left (going down-left)
            # diag2 shifts right (going down-right)
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return board
    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
