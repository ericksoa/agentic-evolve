"""
N-Queens Solver - Gen1 Variant C: Bitmask Diagonals

Mutation: Replace boolean arrays with integer bitmasks for diagonal tracking.
Uses bitwise operations instead of array indexing for faster conflict checks.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using permutation approach with bitmask diagonal tracking.

    Uses integer bitmasks instead of boolean arrays:
    - Single bitwise AND to check if diagonal is free
    - Single bitwise OR to mark diagonal as used
    - Single bitwise XOR to unmark diagonal
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = list(range(n))  # Start with identity permutation

    # Use integer bitmasks for diagonals
    diag1_mask = 0  # row - col + n - 1
    diag2_mask = 0  # row + col

    def backtrack(row: int, d1_mask: int, d2_mask: int) -> bool:
        if row == n:
            return True

        # Try swapping current position with each remaining position
        for i in range(row, n):
            col = solution[i]
            d1_bit = 1 << (row - col + n - 1)
            d2_bit = 1 << (row + col)

            # Check if both diagonals are free using bitwise AND
            if not (d1_mask & d1_bit) and not (d2_mask & d2_bit):
                # Swap to put this column at current row
                solution[row], solution[i] = solution[i], solution[row]

                # Mark diagonals using bitwise OR
                if backtrack(row + 1, d1_mask | d1_bit, d2_mask | d2_bit):
                    return True

                # Undo swap (no need to undo masks - they're passed by value)
                solution[row], solution[i] = solution[i], solution[row]

        return False

    if backtrack(0, 0, 0):
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
