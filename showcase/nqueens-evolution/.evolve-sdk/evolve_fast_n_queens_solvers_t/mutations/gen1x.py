"""
Hybrid Crossover: Permutation-based with Bitwise Diagonal Tracking
Combines:
- gen0_a: Bitwise operations for efficient diagonal conflict detection
- gen0_d: Permutation-based approach (column uniqueness guaranteed)
- gen0_e: Incremental conflict checking during placement
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using permutation approach with bitwise diagonal tracking."""
    board = list(range(n))  # Start with identity permutation

    def backtrack(pos: int, diag1: int, diag2: int) -> bool:
        """
        Backtrack with bitwise diagonal tracking.
        pos: current row being placed
        diag1: bitmask of occupied \ diagonals (row - col + n - 1)
        diag2: bitmask of occupied / diagonals (row + col)
        """
        if pos == n:
            return True

        for i in range(pos, n):
            col = board[i]

            # Calculate diagonal indices for this position
            d1_bit = 1 << (pos - col + n - 1)
            d2_bit = 1 << (pos + col)

            # Check if diagonals are free using bitwise AND
            if not (diag1 & d1_bit) and not (diag2 & d2_bit):
                # Swap to place queen
                board[pos], board[i] = board[i], board[pos]

                # Recurse with updated diagonal masks
                if backtrack(pos + 1, diag1 | d1_bit, diag2 | d2_bit):
                    return True

                # Swap back (restore)
                board[pos], board[i] = board[i], board[pos]

        return False

    if backtrack(0, 0, 0):
        return board
    return None


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
