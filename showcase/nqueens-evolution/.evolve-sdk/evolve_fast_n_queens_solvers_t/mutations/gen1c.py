"""
Variant gen1c: Permutation-based with Incremental Diagonal Tracking
Uses sets to track diagonals instead of O(n²) pairwise checking.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using permutation approach with O(1) diagonal checks."""
    board = list(range(n))  # Start with identity permutation
    diag1 = set()  # row - col diagonals
    diag2 = set()  # row + col diagonals

    def backtrack(pos: int) -> bool:
        if pos == n:
            return True

        for i in range(pos, n):
            col = board[i]
            d1 = pos - col
            d2 = pos + col

            # Check if diagonals are free
            if d1 not in diag1 and d2 not in diag2:
                # Swap
                board[pos], board[i] = board[i], board[pos]
                diag1.add(d1)
                diag2.add(d2)

                if backtrack(pos + 1):
                    return True

                # Backtrack
                diag1.remove(d1)
                diag2.remove(d2)
                board[pos], board[i] = board[i], board[pos]

        return False

    if backtrack(0):
        return board
    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
