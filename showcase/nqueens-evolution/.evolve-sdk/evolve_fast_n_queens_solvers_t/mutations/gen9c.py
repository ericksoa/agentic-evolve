"""
N-Queens Solver - Gen9c: Pop Count Optimization with bit_count()

Parent: gen1a (fitness 18722.02)

Mutation: Replace bin(x).count('1') and len(available) with Python's native
int.bit_count() method for counting available positions.

Key insight: The parent builds a list of available columns in the MRV inner loop,
then counts with len(). By using bitmasks and int.bit_count() (Python 3.10+),
we can:
1. Compute availability with O(1) bitwise operations
2. Count available positions with O(1) bit_count()
3. Only build the list when we've found the minimum row

This reduces memory allocations and uses the CPU's native POPCNT instruction.

The mutation also uses a lazy list construction - we only build the list of
available columns for the selected minimum row, avoiding list allocations
during the counting phase.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using MRV with bitmask counting optimization.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n
    all_cols = (1 << n) - 1

    # Track conflicts using arrays (same as parent)
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)  # row - col + n - 1
    diag2_used = [False] * (2 * n - 1)  # row + col

    row_done = [False] * n
    n_minus_1 = n - 1

    # Precompute bitmasks for fast availability checking
    # For each (row, col), precompute d1 and d2 indices
    d1_idx = [[row - col + n_minus_1 for col in range(n)] for row in range(n)]
    d2_idx = [[row + col for col in range(n)] for row in range(n)]

    def count_available_fast(row: int) -> int:
        """Count available columns for a row using bit manipulation."""
        mask = 0
        d1_row = d1_idx[row]
        d2_row = d2_idx[row]
        for col in range(n):
            if not col_used[col] and not diag1_used[d1_row[col]] and not diag2_used[d2_row[col]]:
                mask |= (1 << col)
        return mask.bit_count(), mask

    def get_columns_from_mask(mask: int) -> list:
        """Extract column indices from bitmask."""
        cols = []
        while mask:
            bit = mask & -mask
            cols.append(bit.bit_length() - 1)
            mask ^= bit
        return cols

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # MRV: select row with fewest available positions
        min_count = n + 1
        min_row = -1
        min_mask = 0

        for row in range(n):
            if row_done[row]:
                continue

            # Count available positions using bitmask
            count, mask = count_available_fast(row)

            if count == 0:
                # Row with no options - immediate fail
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_mask = mask
                if count == 1:
                    break

        if min_row == -1:
            return False

        row = min_row
        row_done[row] = True
        base_d1 = row + n_minus_1

        # Only now extract the column list from the mask
        available = get_columns_from_mask(min_mask)

        for col in available:
            d1 = base_d1 - col
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
