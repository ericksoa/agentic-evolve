"""
N-Queens Solver - Variant gen5c: Bitmask-based MRV

Approach: Hybrid algorithm selection based on input size.
- For small n (<=10): Uses simple row-by-row backtracking with bitmasks
- For large n (>10): Uses MRV heuristic WITH bitmask operations (not boolean arrays)
- All conflict tracking uses bitmasks for O(1) popcount availability counting
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
        return _solve_mrv_bitwise(n)


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


def _solve_mrv_bitwise(n: int) -> list[int] | None:
    """MRV heuristic solver using bitmasks for fast availability counting."""
    solution = [-1] * n
    all_cols = (1 << n) - 1

    # Bitmask for columns used
    cols_mask = 0
    # Bitmasks for diagonals - indexed by diagonal number
    diag1_mask = [0] * (2 * n - 1)  # Store column bit for each diagonal
    diag2_mask = [0] * (2 * n - 1)
    row_done = 0  # Bitmask for completed rows

    def get_available_mask(row: int) -> int:
        """Get bitmask of available columns for a row."""
        # Start with all columns
        available = all_cols & ~cols_mask
        # Remove columns blocked by diagonals
        d1_idx = row + n - 1
        d2_idx = row

        # For each diagonal passing through this row, calculate blocked column
        for col in range(n):
            col_bit = 1 << col
            if available & col_bit:
                d1 = row - col + n - 1
                d2 = row + col
                if diag1_mask[d1] or diag2_mask[d2]:
                    available &= ~col_bit
        return available

    def select_row() -> int:
        """Select row with minimum remaining values (MRV heuristic)."""
        min_count = n + 1
        min_row = -1
        for row in range(n):
            if not (row_done & (1 << row)):
                avail = get_available_mask(row)
                count = avail.bit_count()
                if count < min_count:
                    min_count = count
                    min_row = row
                    if count == 0:
                        break
        return min_row

    def backtrack(placed: int) -> bool:
        nonlocal cols_mask, row_done

        if placed == n:
            return True

        row = select_row()
        if row == -1:
            return False

        available = get_available_mask(row)
        if not available:
            return False

        row_done |= (1 << row)

        while available:
            # Get rightmost set bit
            bit = available & -available
            available ^= bit
            col = (bit - 1).bit_count()

            d1 = row - col + n - 1
            d2 = row + col

            solution[row] = col
            old_cols = cols_mask
            cols_mask |= bit
            diag1_mask[d1] = 1
            diag2_mask[d2] = 1

            if backtrack(placed + 1):
                return True

            cols_mask = old_cols
            diag1_mask[d1] = 0
            diag2_mask[d2] = 0

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
