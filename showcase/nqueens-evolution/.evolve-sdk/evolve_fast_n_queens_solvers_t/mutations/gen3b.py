"""
N-Queens Solver - Variant gen3b: Fully Bitmask MRV with Optimized Dispatch

Approach: Hybrid algorithm selection based on input size.
- For small n (<=10): Uses simple row-by-row backtracking with bitmasks
- For large n (>10): Uses MRV heuristic with FULL bitmask operations
- Mutation: Convert MRV's boolean arrays to bitmasks for faster conflict checking
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach based on problem size.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Use simple bitwise backtracking for small n (faster due to lower overhead)
    if n <= 10:
        return _solve_simple_bitwise(n)
    else:
        return _solve_mrv_bitmask(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking."""
    solution = [-1] * n
    all_cols = (1 << n) - 1  # Mask of all columns

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        # Available positions: columns not attacked
        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            # Get rightmost set bit (a valid column)
            bit = available & -available
            available ^= bit  # Remove this bit

            # Convert bit position to column index
            col = (bit - 1).bit_count()
            solution[row] = col

            # Recurse: shift diagonals for next row
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_bitmask(n: int) -> list[int] | None:
    """MRV heuristic solver with full bitmask operations for larger boards."""
    solution = [-1] * n

    # Track conflicts using bitmasks - much faster than boolean arrays
    col_mask = 0
    # For diagonals, we use arrays of bits - but we'll precompute positions
    diag1_mask = 0  # row - col + n - 1 diagonal
    diag2_mask = 0  # row + col diagonal
    row_done_mask = 0

    n_minus_1 = n - 1
    all_rows = (1 << n) - 1

    def backtrack(placed: int) -> bool:
        nonlocal col_mask, diag1_mask, diag2_mask, row_done_mask

        if placed == n:
            return True

        # MRV: select most constrained row
        min_count = n + 1
        min_row = -1
        min_available = None

        # Iterate through unplaced rows
        remaining = all_rows & ~row_done_mask
        while remaining:
            row_bit = remaining & -remaining
            row = row_bit.bit_length() - 1
            remaining &= remaining - 1

            # Build available list for this row using bitmask checks
            available = []
            base_d1 = row + n_minus_1
            d1_shift = base_d1
            d2_base = row

            for col in range(n):
                col_bit = 1 << col
                d1_bit = 1 << (base_d1 - col)
                d2_bit = 1 << (row + col)

                if not (col_mask & col_bit) and not (diag1_mask & d1_bit) and not (diag2_mask & d2_bit):
                    available.append(col)

            count = len(available)
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
        row_done_mask |= 1 << row
        base_d1 = row + n_minus_1

        for col in min_available:
            col_bit = 1 << col
            d1_bit = 1 << (base_d1 - col)
            d2_bit = 1 << (row + col)

            solution[row] = col
            col_mask |= col_bit
            diag1_mask |= d1_bit
            diag2_mask |= d2_bit

            if backtrack(placed + 1):
                return True

            col_mask &= ~col_bit
            diag1_mask &= ~d1_bit
            diag2_mask &= ~d2_bit

        solution[row] = -1
        row_done_mask &= ~(1 << row)
        return False

    if backtrack(0):
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
