"""
Variant gen2b: Permutation-based with Inline Diagonal Check
Mutation: Eliminate set operations entirely, use inline integer-based conflict check.
Parent: gen1c.py (permutation + sets, fitness 197.45)
"""


def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using permutation approach with inline conflict checks."""
    if n <= 0:
        return None
    if n == 1:
        return [0]

    board = list(range(n))  # Start with identity permutation
    # Track diagonal values directly in arrays for fast lookup
    diag1 = [False] * (2 * n - 1)  # row - col + (n-1), range 0 to 2n-2
    diag2 = [False] * (2 * n - 1)  # row + col, range 0 to 2n-2

    def backtrack(pos: int) -> bool:
        if pos == n:
            return True

        for i in range(pos, n):
            col = board[i]
            d1 = pos - col + n - 1
            d2 = pos + col

            # Check if diagonals are free using direct array indexing
            if not diag1[d1] and not diag2[d2]:
                # Swap
                board[pos], board[i] = board[i], board[pos]
                diag1[d1] = True
                diag2[d2] = True

                if backtrack(pos + 1):
                    return True

                # Backtrack
                diag1[d1] = False
                diag2[d2] = False
                board[pos], board[i] = board[i], board[pos]

        return False

    if backtrack(0):
        return board
    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
