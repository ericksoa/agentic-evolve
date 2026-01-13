"""
N-Queens Solver - Gen2a: Bitwise MRV with Lookup Table

Mutation: Convert MRV heuristic from boolean arrays to bitmask operations.
- Uses bitmasks for column and diagonal tracking
- Uses bit_count() for O(1) counting of available positions
- Precomputes diagonal bit patterns for each (row, col) pair
- Uses lookup table to avoid per-iteration bit calculations
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using bitwise MRV heuristic with precomputed lookups.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n

    # Precompute diagonal bit patterns for all (row, col) combinations
    # d1_bits[row][col] = bit position for row-col diagonal
    # d2_bits[row][col] = bit position for row+col diagonal
    d1_bits = [[1 << (row - col + n - 1) for col in range(n)] for row in range(n)]
    d2_bits = [[1 << (row + col) for col in range(n)] for row in range(n)]
    col_bits = [1 << col for col in range(n)]

    # Bitmasks for tracking conflicts
    col_mask = 0
    diag1_mask = 0
    diag2_mask = 0

    row_done = [False] * n

    def backtrack(placed: int) -> bool:
        nonlocal col_mask, diag1_mask, diag2_mask

        if placed == n:
            return True

        # MRV: select most constrained row using bit_count()
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            # Build available list for this row
            row_d1 = d1_bits[row]
            row_d2 = d2_bits[row]
            available = []
            for col in range(n):
                if not (col_mask & col_bits[col]) and not (diag1_mask & row_d1[col]) and not (diag2_mask & row_d2[col]):
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
        row_done[row] = True
        row_d1 = d1_bits[row]
        row_d2 = d2_bits[row]

        for col in min_available:
            col_bit = col_bits[col]
            d1_bit = row_d1[col]
            d2_bit = row_d2[col]

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
