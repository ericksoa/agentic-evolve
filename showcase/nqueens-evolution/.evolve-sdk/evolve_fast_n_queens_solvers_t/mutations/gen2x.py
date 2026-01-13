"""
N-Queens Solver - Gen2x: Bitwise MRV Hybrid

Crossover: Combines bitwise operations from gen0_a with MRV heuristic from gen1a.
- Uses bitmasks for O(1) conflict detection (from gen0_a)
- Uses MRV heuristic to select most constrained row (from gen1a)
- Inlines MRV selection to avoid function call overhead (from gen1a)
- Uses popcount for fast counting of available positions
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using bitwise MRV heuristic.

    Combines the search-space reduction of MRV with the
    speed of bitwise conflict detection.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n

    # Bitmask for all columns (used to mask out irrelevant bits)
    all_cols = (1 << n) - 1

    # Track conflicts using bitmasks per row
    # For each row, we track what's blocked by cols, diag1, diag2
    # We store the current column conflicts (shared across all rows)
    cols = 0

    # For diagonals, we need per-row shifted versions
    # diag1[row] blocks are: (base_diag1) >> row (shifted right)
    # diag2[row] blocks are: (base_diag2) << row (shifted left)
    # We track the "base" values and compute per-row on the fly

    # Actually, for MRV we need to compute available positions per row
    # Let's track: diag1_blocks and diag2_blocks as base values
    # For row r, diag1 blocking is: diag1_base >> (n-1-r) ... this is complex

    # Simpler approach: track diagonal conflicts with arrays but use bitmask for cols
    # and use bit_count for fast MRV counting

    # Track which rows are done
    row_done = 0  # Bitmask of completed rows

    # For diagonal tracking, we use arrays but with fast iteration
    diag1_used = [False] * (2 * n - 1)  # row - col + n - 1
    diag2_used = [False] * (2 * n - 1)  # row + col

    n_minus_1 = n - 1

    def get_available_mask(row: int) -> int:
        """Get bitmask of available columns for a row using bitwise ops."""
        # Start with columns not used
        avail = all_cols & ~cols

        # Remove diagonally blocked columns
        base_d1 = row + n_minus_1
        for col in range(n):
            if avail & (1 << col):
                if diag1_used[base_d1 - col] or diag2_used[row + col]:
                    avail &= ~(1 << col)
        return avail

    def backtrack(placed: int) -> bool:
        nonlocal cols, row_done

        if placed == n:
            return True

        # Inlined MRV: select most constrained row using bit_count
        min_count = n + 1
        min_row = -1
        min_available = 0

        for row in range(n):
            if row_done & (1 << row):
                continue

            # Get available positions as bitmask
            avail = all_cols & ~cols
            base_d1 = row + n_minus_1

            # Check diagonals - build mask of blocked cols
            for col in range(n):
                if avail & (1 << col):
                    if diag1_used[base_d1 - col] or diag2_used[row + col]:
                        avail &= ~(1 << col)

            count = bin(avail).count('1')  # popcount

            if count == 0:
                # Row with no options - immediate fail
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_available = avail
                if count == 1:
                    break  # Can't do better than 1

        if min_row == -1:
            return False

        row = min_row
        row_done |= (1 << row)
        base_d1 = row + n_minus_1

        # Iterate through available columns using bit manipulation
        available = min_available
        while available:
            col_bit = available & -available  # Rightmost set bit
            col = col_bit.bit_length() - 1

            d1 = base_d1 - col
            d2 = row + col

            solution[row] = col
            cols |= col_bit
            diag1_used[d1] = True
            diag2_used[d2] = True

            if backtrack(placed + 1):
                return True

            cols &= ~col_bit
            diag1_used[d1] = False
            diag2_used[d2] = False

            available &= available - 1  # Remove rightmost bit

        solution[row] = -1
        row_done &= ~(1 << row)
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
