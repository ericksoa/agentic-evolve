"""
N-Queens Solver - Gen7c: Incremental Bitmask MRV

Mutation: Maintain per-row availability bitmasks that are incrementally updated
when queens are placed/removed. Instead of recomputing available columns for
each row from scratch during MRV selection, we maintain row_available[row]
masks that get updated when constraints change. This reduces the MRV selection
from O(n^2) to O(n) since we just need to scan counts, not recompute them.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using incrementally-updated availability bitmasks.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n
    all_cols = (1 << n) - 1

    # Precompute column masks for each diagonal
    # For each diagonal index, which columns does it cover for each row?
    n_minus_1 = n - 1
    diag_size = 2 * n - 1

    # d1_col[row][d1_idx] = column on this diagonal at this row (or -1)
    # d2_col[row][d2_idx] = column on this diagonal at this row (or -1)
    d1_col = [[-1] * diag_size for _ in range(n)]
    d2_col = [[-1] * diag_size for _ in range(n)]

    for row in range(n):
        for col in range(n):
            d1 = row - col + n_minus_1
            d2 = row + col
            d1_col[row][d1] = col
            d2_col[row][d2] = col

    # Track which rows are done
    row_done = [False] * n

    # Track which columns/diagonals are blocked (as bitmasks for fast updates)
    col_mask = 0  # bit set = column blocked
    diag1_mask = [0] * diag_size  # For each d1, bitmask of rows affected
    diag2_mask = [0] * diag_size  # For each d2, bitmask of rows affected

    # Per-row availability: bitmask of available columns
    # Updated incrementally as queens are placed/removed
    row_available = [all_cols] * n
    row_avail_count = [n] * n  # Cache of bit_count for each row

    def place_queen(row: int, col: int):
        """Place queen and update all affected rows' availability."""
        nonlocal col_mask

        d1 = row - col + n_minus_1
        d2 = row + col

        col_bit = 1 << col
        col_mask |= col_bit

        # Update availability for all unfinished rows
        for r in range(n):
            if row_done[r]:
                continue
            old = row_available[r]
            # Remove column
            new = old & ~col_bit
            # Remove diagonal conflicts
            d1_c = d1_col[r][d1]
            d2_c = d2_col[r][d2]
            if d1_c >= 0:
                new &= ~(1 << d1_c)
            if d2_c >= 0:
                new &= ~(1 << d2_c)
            if new != old:
                row_available[r] = new
                row_avail_count[r] = new.bit_count()

    def remove_queen(row: int, col: int):
        """Remove queen and restore affected rows' availability."""
        nonlocal col_mask

        d1 = row - col + n_minus_1
        d2 = row + col

        col_bit = 1 << col
        col_mask &= ~col_bit

        # Restore availability for all unfinished rows
        for r in range(n):
            if row_done[r]:
                continue
            old = row_available[r]
            new = old

            # Restore column if not blocked by another queen
            if not (col_mask & col_bit):
                new |= col_bit

            # Restore diagonal if not blocked
            d1_c = d1_col[r][d1]
            d2_c = d2_col[r][d2]
            if d1_c >= 0:
                d1_bit = 1 << d1_c
                # Check if still blocked by col_mask or diagonals
                # Simplified: just add back, will be correct due to other constraints
                new |= d1_bit
            if d2_c >= 0:
                d2_bit = 1 << d2_c
                new |= d2_bit

            # Recompute from scratch for correctness (incremental is complex)
            actual = all_cols & ~col_mask
            for c in range(n):
                r_d1 = r - c + n_minus_1
                r_d2 = r + c
                # Check if any placed queen blocks this
                for placed_row in range(n):
                    if solution[placed_row] >= 0 and placed_row != row:
                        pc = solution[placed_row]
                        if pc == c:
                            actual &= ~(1 << c)
                        pd1 = placed_row - pc + n_minus_1
                        pd2 = placed_row + pc
                        if r_d1 == pd1:
                            actual &= ~(1 << c)
                        if r_d2 == pd2:
                            actual &= ~(1 << c)
            row_available[r] = actual
            row_avail_count[r] = actual.bit_count()

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # MRV: find row with minimum available count (O(n) scan)
        min_count = n + 1
        min_row = -1

        for row in range(n):
            if row_done[row]:
                continue
            count = row_avail_count[row]
            if count == 0:
                return False
            if count < min_count:
                min_count = count
                min_row = row
                if count == 1:
                    break

        if min_row == -1:
            return False

        row = min_row
        row_done[row] = True
        available = row_available[row]

        # Iterate through available columns
        while available:
            col_bit = available & -available
            col = col_bit.bit_length() - 1
            available &= available - 1

            solution[row] = col
            place_queen(row, col)

            if backtrack(placed + 1):
                return True

            remove_queen(row, col)

        solution[row] = -1
        row_done[row] = False
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
