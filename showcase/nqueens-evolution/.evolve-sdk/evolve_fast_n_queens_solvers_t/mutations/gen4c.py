"""
N-Queens Solver - Variant gen4c: Optimized Bit Position Extraction

Approach: Replaces (bit - 1).bit_count() with bit.bit_length() - 1
for faster column index extraction from bitmask positions.
bit_length() is typically faster than bit_count() as it doesn't
need to count all set bits, just find the highest one.
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
        return _solve_mrv(n)


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

            # Convert bit position to column index using bit_length
            # bit_length() - 1 is faster than (bit-1).bit_count()
            col = bit.bit_length() - 1
            solution[row] = col

            # Recurse: shift diagonals for next row
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv(n: int) -> list[int] | None:
    """MRV heuristic solver for larger boards."""
    solution = [-1] * n

    # Track conflicts using arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)
    row_done = [False] * n

    def count_available(row: int) -> int:
        count = 0
        for col in range(n):
            if not col_used[col] and not diag1_used[row - col + n - 1] and not diag2_used[row + col]:
                count += 1
        return count

    def get_available(row: int) -> list[int]:
        result = []
        for col in range(n):
            if not col_used[col] and not diag1_used[row - col + n - 1] and not diag2_used[row + col]:
                result.append(col)
        return result

    def select_row() -> int:
        min_count = n + 1
        min_row = -1
        for row in range(n):
            if not row_done[row]:
                count = count_available(row)
                if count < min_count:
                    min_count = count
                    min_row = row
                    if count == 0:
                        break
        return min_row

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        row = select_row()
        if row == -1:
            return False

        available = get_available(row)
        if not available:
            return False

        row_done[row] = True

        for col in available:
            d1 = row - col + n - 1
            d2 = row + col

            solution[row] = col
            col_used[col] = True
            diag1_used[d1] = True
            diag2_used[d2] = True

            if backtrack(placed + 1):
                return True

            col_used[col] = False
            diag1_used[d1] = False
            diag2_used[d2] = False

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
