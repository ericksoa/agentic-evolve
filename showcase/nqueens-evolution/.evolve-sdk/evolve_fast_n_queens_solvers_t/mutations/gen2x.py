"""
Hybrid gen2x: Iterative Bitwise Backtracking with Stack-based State
Combines:
- gen1b: Pure bitwise backtracking with bit manipulation (best fitness)
- Iterative approach: Eliminates recursive function call overhead
- Stack-based state: Maintains available positions for efficient backtracking
"""


def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using iterative bitwise backtracking."""
    if n <= 0:
        return None
    if n == 1:
        return [0]

    board = [-1] * n
    all_ones = (1 << n) - 1  # Mask with n bits set

    # Stack stores: (row, cols, diag1, diag2, available)
    # available tracks remaining positions to try for current row
    row = 0
    cols = 0
    diag1 = 0
    diag2 = 0
    available = all_ones  # All positions available for row 0

    # Stack for backtracking - stores state before each placement
    stack = []

    while True:
        if available:
            # Get rightmost available position
            bit = available & (-available)
            available &= available - 1  # Remove this bit for next iteration

            # Convert bit position to column index
            col = (bit - 1).bit_count()
            board[row] = col

            # Save current state before advancing
            stack.append((row, cols, diag1, diag2, available))

            # Advance to next row with updated constraints
            cols |= bit
            diag1 = (diag1 | bit) << 1
            diag2 = (diag2 | bit) >> 1
            row += 1

            if row == n:
                return board  # Solution found!

            # Calculate available positions for new row
            available = all_ones & ~(cols | diag1 | diag2)
        else:
            # No available positions - backtrack
            if not stack:
                return None  # No solution

            # Restore state from stack
            row, cols, diag1, diag2, available = stack.pop()


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
