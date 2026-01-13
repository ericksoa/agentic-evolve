"""
Mutation: Precomputed Lookup Tables for Diagonal Bits
Parent: gen1x.py
Change: Precompute all diagonal bit masks in lookup tables to eliminate
bitshift operations (1 << x) from the hot inner loop.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using permutation approach with precomputed diagonal lookup tables."""
    board = list(range(n))  # Start with identity permutation

    # Precompute lookup tables for diagonal bit masks
    # d1_table[row][col] = 1 << (row - col + n - 1)
    # d2_table[row][col] = 1 << (row + col)
    d1_table = [[1 << (row - col + n - 1) for col in range(n)] for row in range(n)]
    d2_table = [[1 << (row + col) for col in range(n)] for row in range(n)]

    def backtrack(pos: int, diag1: int, diag2: int) -> bool:
        """
        Backtrack with bitwise diagonal tracking using precomputed tables.
        pos: current row being placed
        diag1: bitmask of occupied \ diagonals
        diag2: bitmask of occupied / diagonals
        """
        if pos == n:
            return True

        # Cache row lookups
        d1_row = d1_table[pos]
        d2_row = d2_table[pos]

        for i in range(pos, n):
            col = board[i]

            # Use precomputed diagonal bits
            d1_bit = d1_row[col]
            d2_bit = d2_row[col]

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
