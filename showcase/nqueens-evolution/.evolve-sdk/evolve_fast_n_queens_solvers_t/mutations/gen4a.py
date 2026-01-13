"""
N-Queens Solver - Gen4a: Propagated Bitmask Backtracking

Mutation: Replace MRV with simpler row-by-row backtracking using propagated
conflict masks. Instead of scanning all rows to find the most constrained,
process rows sequentially and propagate diagonal conflicts via bit shifting.
- Eliminates O(n) MRV selection loop per placement
- Uses bitwise shift to propagate diagonal constraints row-to-row
- Simpler control flow with less overhead for moderate n values
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using propagated bitmask backtracking (row-by-row).
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n
    all_cols = (1 << n) - 1  # Bitmask with n bits set

    def backtrack(row: int, col_mask: int, d1_mask: int, d2_mask: int) -> bool:
        """
        row: current row to place queen
        col_mask: columns already occupied
        d1_mask: diagonal / attacks reaching this row (shifted down)
        d2_mask: diagonal \ attacks reaching this row (shifted down)
        """
        if row == n:
            return True

        # Available positions = all columns minus (columns | shifted diagonals)
        blocked = col_mask | d1_mask | d2_mask
        available = all_cols & ~blocked

        if not available:
            return False

        # Try each available column
        while available:
            col_bit = available & -available  # Lowest set bit
            col = col_bit.bit_length() - 1
            available &= available - 1  # Clear lowest bit

            solution[row] = col

            # Propagate constraints to next row:
            # - Column stays the same
            # - d1 (/) shifts left as we go down
            # - d2 (\) shifts right as we go down
            new_d1 = (d1_mask | col_bit) << 1
            new_d2 = (d2_mask | col_bit) >> 1

            if backtrack(row + 1, col_mask | col_bit, new_d1, new_d2):
                return True

        solution[row] = -1
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
