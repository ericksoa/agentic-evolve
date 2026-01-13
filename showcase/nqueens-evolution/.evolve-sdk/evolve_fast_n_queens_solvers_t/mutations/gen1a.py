"""
Variant gen1a: Bitwise Backtracking with First Row Symmetry Breaking
Exploits symmetry: only try first half of columns in row 0.
This roughly halves the search space for finding first solution.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using bitwise backtracking with symmetry breaking."""
    if n == 0:
        return []
    if n == 1:
        return [0]

    board = [-1] * n
    all_cols = (1 << n) - 1

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

    # Symmetry breaking: only try first half of columns in row 0
    # For odd n, include middle column
    half = (n + 1) // 2
    first_row_mask = (1 << half) - 1  # Only columns 0 to half-1

    available = first_row_mask

    while available:
        pos = available & -available
        available ^= pos

        col = pos.bit_length() - 1
        board[0] = col

        if backtrack(1, pos, pos << 1, pos >> 1):
            return board

        board[0] = -1

    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
