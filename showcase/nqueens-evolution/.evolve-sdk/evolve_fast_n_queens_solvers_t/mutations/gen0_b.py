"""
Variant B: Iterative Backtracking with Set-based Constraints
Uses sets for O(1) constraint checking with an iterative stack-based approach.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using iterative backtracking with sets."""
    board = [-1] * n
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    row = 0
    col = 0

    while row >= 0:
        # Try to place queen in current row
        found = False

        while col < n:
            d1 = row - col
            d2 = row + col

            if col not in cols and d1 not in diag1 and d2 not in diag2:
                # Place queen
                board[row] = col
                cols.add(col)
                diag1.add(d1)
                diag2.add(d2)
                found = True
                row += 1
                col = 0
                break
            col += 1

        if not found:
            # Backtrack
            row -= 1
            if row >= 0:
                old_col = board[row]
                cols.discard(old_col)
                diag1.discard(row - old_col)
                diag2.discard(row + old_col)
                col = old_col + 1

        if row == n:
            return board

    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
