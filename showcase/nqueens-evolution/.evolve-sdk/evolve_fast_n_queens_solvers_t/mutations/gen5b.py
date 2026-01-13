"""
N-Queens Solver - Gen5b: Optimized bit-to-column conversion

Mutation from gen3x:
- Changed (bit - 1).bit_count() to bit.bit_length() - 1 for column index calculation
- bit_length() is a single CPU instruction (CLZ/BSR) vs bit_count() which counts all bits
- This reduces the critical path in the inner loop of backtracking
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
        return _solve_mrv_inlined(n)


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

            # MUTATION: Use bit_length() - 1 instead of (bit-1).bit_count()
            # bit_length() returns position + 1 for the highest set bit
            # For a single bit set at position k, bit_length() = k + 1
            col = bit.bit_length() - 1
            solution[row] = col

            # Recurse: shift diagonals for next row
            if backtrack(row + 1, cols | bit, (diag1 | bit) << 1, (diag2 | bit) >> 1):
                return True

        return False

    if backtrack(0, 0, 0, 0):
        return solution
    return None


def _solve_mrv_inlined(n: int) -> list[int] | None:
    """
    MRV solver with inlined heuristic (from gen1a) and precomputed lookups (from gen2a).
    """
    solution = [-1] * n

    # From gen2a: Precompute bit patterns for faster operations
    d1_bits = [[1 << (row - col + n - 1) for col in range(n)] for row in range(n)]
    d2_bits = [[1 << (row + col) for col in range(n)] for row in range(n)]
    col_bits = [1 << col for col in range(n)]

    # Bitmasks for tracking conflicts (from gen2a)
    col_mask = 0
    diag1_mask = 0
    diag2_mask = 0

    row_done = [False] * n

    def backtrack(placed: int) -> bool:
        nonlocal col_mask, diag1_mask, diag2_mask

        if placed == n:
            return True

        # From gen1a: Inlined MRV selection with early termination
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            # Inlined availability check using precomputed lookups (gen2a style)
            row_d1 = d1_bits[row]
            row_d2 = d2_bits[row]
            available = []
            for col in range(n):
                if not (col_mask & col_bits[col]) and not (diag1_mask & row_d1[col]) and not (diag2_mask & row_d2[col]):
                    available.append(col)

            count = len(available)

            # From gen1a: Immediate fail when row has no options
            if count == 0:
                return False

            if count < min_count:
                min_count = count
                min_row = row
                min_available = available
                # From gen1a: Early termination when count == 1
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
