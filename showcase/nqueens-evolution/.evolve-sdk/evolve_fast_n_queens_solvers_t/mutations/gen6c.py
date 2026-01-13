"""
N-Queens Solver - Gen6c: Bitwise MRV with direct column masks

Mutation from gen5b:
- Changed diagonal tracking in MRV solver from per-cell lookups to direct column masks
- Instead of d1_bits[row][col] storing a diagonal identifier, we use shifted bitmasks
- For each row, diag1_row[r] = (base_diag1 >> (n-1-r)) & all_cols gives attacked columns
- diag2_row[r] = (base_diag2 >> r) & all_cols gives attacked columns
- This eliminates the inner column loop in MRV selection
- Availability is now: all_cols & ~(col_mask | diag1_row | diag2_row)
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using hybrid approach with optimized dispatch.
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


def _solve_mrv_bitwise(n: int) -> list[int] | None:
    """
    MRV solver with bitwise availability calculation.

    MUTATION: Use shifted bitmasks to compute available columns directly.

    Key insight: In the simple bitwise solver, diagonals are tracked as shifted
    bitmasks relative to the current row. For MRV, we need to track them
    relative to ALL rows simultaneously.

    We store diagonal attacks at a reference row (row 0), and shift to compute
    attacks at any other row:
    - diag1 (row-col=const): shifts left as row increases
    - diag2 (row+col=const): shifts right as row increases
    """
    solution = [-1] * n
    all_cols = (1 << n) - 1
    wide_mask = (1 << (2 * n)) - 1  # For wider diagonal tracking

    col_mask = 0
    # Track diagonals at row 0 reference point (using wider bitmask to avoid overflow)
    diag1_base = 0  # row - col constant: to get attacks at row r, shift right by r
    diag2_base = 0  # row + col constant: to get attacks at row r, shift right by r

    row_done = [False] * n

    def get_available(row: int) -> int:
        """Get available columns for a row as a bitmask."""
        # Shift diag1 to get attacks at this row
        # When queen at (qr, qc), diag1 attack at row r is at col qc + (r - qr)
        # We store diag1_base such that for row 0: col qc - qr (shifted by n-1 to keep positive)
        # At row r: shift left by r to get col qc - qr + r = qc + (r - qr)
        d1 = (diag1_base >> (n - 1 - row)) & all_cols

        # Shift diag2 to get attacks at this row
        # When queen at (qr, qc), diag2 attack at row r is at col qc - (r - qr) = qc + qr - r
        # We store diag2_base such that for row 0: col qc + qr
        # At row r: shift right by r to get col qc + qr - r
        d2 = (diag2_base >> row) & all_cols

        return all_cols & ~(col_mask | d1 | d2)

    def place_queen(row: int, col: int) -> None:
        """Place a queen and update attack masks."""
        nonlocal col_mask, diag1_base, diag2_base
        col_mask |= (1 << col)
        # diag1: at row 0, this diagonal attacks col (col - row)
        # We shift by (n-1) to keep all values positive in the bitmask
        # At row 0: col - row + shift = col - row + (n-1)
        diag1_base |= (1 << (col - row + n - 1))
        # diag2: at row 0, this diagonal attacks col (col + row)
        diag2_base |= (1 << (col + row))

    def remove_queen(row: int, col: int) -> None:
        """Remove a queen and update attack masks."""
        nonlocal col_mask, diag1_base, diag2_base
        col_mask &= ~(1 << col)
        diag1_base &= ~(1 << (col - row + n - 1))
        diag2_base &= ~(1 << (col + row))

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # MRV selection with O(1) availability per row
        min_count = n + 1
        min_row = -1
        min_available = 0

        for row in range(n):
            if row_done[row]:
                continue

            available = get_available(row)
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
        row_done[row] = True
        available = min_available

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
