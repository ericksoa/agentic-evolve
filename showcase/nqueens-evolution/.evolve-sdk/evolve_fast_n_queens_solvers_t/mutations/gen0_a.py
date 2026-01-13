"""
N-Queens Solver - Variant A: Bitwise Backtracking with Bitmasks

Approach: Uses bitwise operations for conflict detection.
- Uses three bitmasks: columns, diagonal1, diagonal2
- Bit manipulation for O(1) conflict checking
- Highly cache-efficient due to integer operations
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using bitwise backtracking.

    Uses bitmasks to track:
    - cols: occupied columns
    - diag1: occupied left-down diagonals (row - col)
    - diag2: occupied right-down diagonals (row + col)
    """
    solution = [-1] * n

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        # Available positions: not in any conflict set
        available = ((1 << n) - 1) & ~(cols | diag1 | diag2)

        while available:
            # Get rightmost set bit (lowest column available)
            col_bit = available & -available
            col = col_bit.bit_length() - 1

            solution[row] = col

            # Try placing queen here
            if backtrack(
                row + 1,
                cols | col_bit,
                (diag1 | col_bit) << 1,
                (diag2 | col_bit) >> 1
            ):
                return True

            # Remove this bit from available
            available &= available - 1

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def verify_solution(n: int, solution: list[int]) -> bool:
    """Verify that a solution is valid."""
    if solution is None or len(solution) != n:
        return False

    for row, col in enumerate(solution):
        if col < 0 or col >= n:
            return False
        for prev_row in range(row):
            prev_col = solution[prev_row]
            if prev_col == col:
                return False
            if abs(prev_col - col) == abs(prev_row - row):
                return False

    return True


if __name__ == "__main__":
    import time
    for n in [8, 12, 16, 20]:
        start = time.perf_counter()
        solution = solve_nqueens(n)
        elapsed = time.perf_counter() - start
        valid = verify_solution(n, solution)
        print(f"N={n}: valid={valid}, time={elapsed*1000:.2f}ms, solution={solution}")
