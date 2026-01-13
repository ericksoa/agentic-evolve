"""
Variant D: Permutation-based with Diagonal Validation
Treats as permutation problem with only diagonal checks needed.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using permutation approach."""
    board = list(range(n))  # Start with identity permutation

    def has_diagonal_conflicts(perm: list[int], limit: int) -> bool:
        """Check for diagonal conflicts up to limit."""
        for i in range(limit):
            for j in range(i + 1, limit):
                if abs(perm[i] - perm[j]) == j - i:
                    return True
        return False

    def backtrack(pos: int) -> bool:
        if pos == n:
            return True

        for i in range(pos, n):
            # Swap
            board[pos], board[i] = board[i], board[pos]

            # Check only new diagonal conflicts
            if not has_diagonal_conflicts(board, pos + 1):
                if backtrack(pos + 1):
                    return True

            # Swap back
            board[pos], board[i] = board[i], board[pos]

        return False

    if backtrack(0):
        return board
    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
