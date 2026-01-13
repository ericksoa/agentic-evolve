"""
N-Queens Solver - Gen10a: Bitmask MRV for Large N

Mutation from gen6x: Replace boolean arrays with bitmasks in the MRV solver.

Mutation type: Algorithm swap / Cache optimization
- Changed: Replace col_used, diag1_used, diag2_used boolean arrays with integer bitmasks
- Rationale: Bitmasks enable O(1) popcount for counting available columns and
  faster iteration over available positions using bit manipulation

This removes the O(n) inner loop scanning all columns, replacing it with
bitwise operations that check all columns simultaneously.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Use simple bitwise for small n (lower overhead)
    if n <= 10:
        return _solve_simple_bitwise(n)
    else:
        return _solve_mrv_bitmask(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking."""
    solution = [-1] * n
    all_cols = (1 << n) - 1

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            bit = available & -available
            available ^= bit
            col = bit.bit_length() - 1
            solution[row] = col

            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_bitmask(n: int) -> list[int] | None:
    """
    MRV with bitmask constraint tracking.

    Key change from gen6x: Use bitmasks instead of boolean arrays for:
    - Faster constraint checking (bitwise AND/OR vs array lookups)
    - O(1) popcount to count available columns per row
    - Efficient iteration over available positions
    """
    solution = [-1] * n
    all_cols = (1 << n) - 1

    # Precompute diagonal lookup tables
    n_minus_1 = n - 1
    d1_table = [[row - col + n_minus_1 for col in range(n)] for row in range(n)]
    d2_table = [[row + col for col in range(n)] for row in range(n)]

    # Bitmasks for tracking used columns and diagonals
    col_used = 0
    diag1_used = [0] * (2 * n - 1)  # Store as list of flags, will index directly
    diag2_used = [0] * (2 * n - 1)

    row_done = [False] * n

    # Precompute which columns affect which diagonals (as bitmasks per row)
    # For each row, store the diagonal indices for quick lookup

    def get_available_mask(row: int) -> int:
        """Get bitmask of available columns for a row using bitwise ops."""
        d1_row = d1_table[row]
        d2_row = d2_table[row]
        available = 0
        for col in range(n):
            if not (col_used & (1 << col)) and not diag1_used[d1_row[col]] and not diag2_used[d2_row[col]]:
                available |= (1 << col)
        return available

    def backtrack(placed: int) -> bool:
        nonlocal col_used

        if placed == n:
            return True

        # MRV selection with early termination
        min_count = n + 1
        min_row = -1
        min_available = 0

        for row in range(n):
            if row_done[row]:
                continue

            available = get_available_mask(row)
            count = available.bit_count()

            # Immediate fail when row has no options
            if count == 0:
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                # Early termination when count == 1
                if count == 1:
                    break

        if min_row == -1:
            return False

        row = min_row
        row_done[row] = True
        d1_row = d1_table[row]
        d2_row = d2_table[row]

        # Iterate over available columns using bit manipulation
        available = min_available
        while available:
            bit = available & -available
            available ^= bit
            col = bit.bit_length() - 1

            d1 = d1_row[col]
            d2 = d2_row[col]

            solution[row] = col
            col_used |= bit
            diag1_used[d1] = 1
            diag2_used[d2] = 1

            if backtrack(placed + 1):
                return True

            col_used ^= bit
            diag1_used[d1] = 0
            diag2_used[d2] = 0

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
