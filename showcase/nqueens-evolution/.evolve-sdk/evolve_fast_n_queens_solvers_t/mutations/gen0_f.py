"""
Variant F: Middle-Out Search with Smart Ordering
Starts from middle rows/columns for faster convergence.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens starting from middle columns."""
    board = [-1] * n
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)
    diag2 = [False] * (2 * n - 1)

    # Order columns from middle outward
    middle = n // 2
    col_order = []
    for i in range(n):
        if i % 2 == 0:
            col_order.append(middle + i // 2)
        else:
            col_order.append(middle - (i + 1) // 2)
    col_order = [c for c in col_order if 0 <= c < n]

    def backtrack(row: int) -> bool:
        if row == n:
            return True

        for col in col_order:
            d1 = row - col + n - 1
            d2 = row + col

            if not cols[col] and not diag1[d1] and not diag2[d2]:
                board[row] = col
                cols[col] = diag1[d1] = diag2[d2] = True

                if backtrack(row + 1):
                    return True

                cols[col] = diag1[d1] = diag2[d2] = False

        return False

    if backtrack(0):
        return board
    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
