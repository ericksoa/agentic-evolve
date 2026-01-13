"""
N-Queens Solver - Gen6b: Bitmask-based conflict tracking

Mutation: Replace boolean array conflict tracking with bitmask integers.
- Use single integers for col_used, diag1_used, diag2_used
- Use bit_count() for fast population counting
- Use bit manipulation to iterate available columns
This eliminates list construction overhead and leverages fast CPU bit ops.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using bitmask-based conflict tracking.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n
    all_cols = (1 << n) - 1  # Mask with all n columns available

    # Precompute diagonal indices (same as parent)
    n_minus_1 = n - 1
    d1_table = [[row - col + n_minus_1 for col in range(n)] for row in range(n)]
    d2_table = [[row + col for col in range(n)] for row in range(n)]

    # Track conflicts using bitmasks instead of arrays
    col_used = 0
    diag1_used = 0
    diag2_used = 0
    row_done = 0

    def backtrack(placed: int, col_used: int, diag1_used: int, diag2_used: int, row_done: int) -> bool:
        if placed == n:
            return True

        # MRV: select most constrained row
        min_count = n + 1
        min_row = -1
        min_available = 0

        for row in range(n):
            if row_done & (1 << row):
                continue

            # Compute available columns using bitmasks
            d1_row = d1_table[row]
            d2_row = d2_table[row]

            # Build mask of blocked columns for this row
            blocked = col_used
            for col in range(n):
                if diag1_used & (1 << d1_row[col]):
                    blocked |= (1 << col)
                if diag2_used & (1 << d2_row[col]):
                    blocked |= (1 << col)

            available = all_cols & ~blocked
            count = available.bit_count()

            if count == 0:
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                if count == 1:
                    break

        if min_row == -1:
            return False

        row = min_row
        new_row_done = row_done | (1 << row)
        d1_row = d1_table[row]
        d2_row = d2_table[row]

        # Iterate through available columns via bit extraction
        avail = min_available
        while avail:
            col = (avail & -avail).bit_length() - 1  # Extract lowest set bit position
            avail &= avail - 1  # Clear lowest set bit

            d1 = d1_row[col]
            d2 = d2_row[col]

            solution[row] = col
            new_col_used = col_used | (1 << col)
            new_diag1_used = diag1_used | (1 << d1)
            new_diag2_used = diag2_used | (1 << d2)

            if backtrack(placed + 1, new_col_used, new_diag1_used, new_diag2_used, new_row_done):
                return True

        solution[row] = -1
        return False

    if backtrack(0, 0, 0, 0, 0):
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
