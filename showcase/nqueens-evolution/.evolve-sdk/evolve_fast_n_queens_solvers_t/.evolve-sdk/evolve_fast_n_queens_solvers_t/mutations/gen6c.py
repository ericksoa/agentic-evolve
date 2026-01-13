"""
N-Queens Solver - Gen6c: Per-row diagonal bitmask tracking for MRV

Mutation from gen5b:
- Changed MRV diagonal tracking from per-cell lookups to per-row bitmask arrays
- Precompute which columns are attacked at each row by existing diagonal placements
- This allows O(1) availability check per row using bitwise AND instead of iterating columns
- Uses popcount (bit_count()) for counting available moves
- Eliminates nested loop in MRV selection
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach with optimized dispatch.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # From gen2b: Use simple bitwise backtracking for small n (faster due to lower overhead)
    if n <= 10:
        return _solve_simple_bitwise(n)
    else:
        return _solve_mrv_bitwise(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking.

    Optimized: Uses bit.bit_length() - 1 instead of (bit - 1).bit_count()
    for faster bit position calculation.
    """
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

            col = bit.bit_length() - 1
            solution[row] = col

            # Recurse: shift diagonals for next row
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_bitwise(n: int) -> list[int] | None:
    """
    MRV solver with per-row bitmask tracking.

    MUTATION: Instead of using precomputed lookup tables for individual cells,
    maintain per-row bitmasks of attacked columns. This allows computing
    available columns for any row with a single bitwise operation.

    For each row r, we track:
    - diag1_attack[r]: columns attacked by diagonal (row-col=const) at row r
    - diag2_attack[r]: columns attacked by anti-diagonal (row+col=const) at row r

    When placing at (row, col), we update all other rows' diagonal attack masks.
    """
    solution = [-1] * n
    all_cols = (1 << n) - 1

    col_mask = 0
    # Per-row bitmasks of columns attacked by diagonals
    diag1_attack = [0] * n  # diagonal attacks (row - col = constant)
    diag2_attack = [0] * n  # anti-diagonal attacks (row + col = constant)

    row_done = [False] * n

    def place_queen(row: int, col: int) -> None:
        """Place a queen and update all attack masks."""
        nonlocal col_mask
        col_mask |= (1 << col)

        # Update diagonal attacks for all other rows
        for r in range(n):
            if r == row:
                continue
            diff = r - row
            # Diag1: row - col = constant, attacking col at row r = col + diff
            d1_col = col + diff
            if 0 <= d1_col < n:
                diag1_attack[r] |= (1 << d1_col)
            # Diag2: row + col = constant, attacking col at row r = col - diff
            d2_col = col - diff
            if 0 <= d2_col < n:
                diag2_attack[r] |= (1 << d2_col)

    def remove_queen(row: int, col: int) -> None:
        """Remove a queen and update all attack masks."""
        nonlocal col_mask
        col_mask &= ~(1 << col)

        # Remove diagonal attacks for all other rows
        for r in range(n):
            if r == row:
                continue
            diff = r - row
            d1_col = col + diff
            if 0 <= d1_col < n:
                diag1_attack[r] &= ~(1 << d1_col)
            d2_col = col - diff
            if 0 <= d2_col < n:
                diag2_attack[r] &= ~(1 << d2_col)

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # MRV selection: find row with minimum available columns
        min_count = n + 1
        min_row = -1
        min_available = 0

        for row in range(n):
            if row_done[row]:
                continue

            # O(1) availability calculation using bitmasks!
            available = all_cols & ~(col_mask | diag1_attack[row] | diag2_attack[row])
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
        available = min_available

        # Try each available column using bitwise extraction
        while available:
            bit = available & -available
            available ^= bit
            col = bit.bit_length() - 1

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
