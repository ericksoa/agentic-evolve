"""
N-Queens Solver - Gen9x: Crossover Hybrid

Combines the best aspects of three high-performing parents:
- gen6x (fitness 20406): Best overall - hybrid dispatch with n<=10 threshold, precomputed tables
- gen8b (fitness 18755): Efficient inline diagonal computation (base_d1 pattern)
- gen1a (fitness 18722): Inlined MRV with explicit min_count==0 fail check

Crossover strategy:
1. Use gen6x's n<=10 threshold for bitwise solver (empirically optimal)
2. Add mid-tier bitwise-MRV for n=11-15 (combining bitwise efficiency with MRV pruning)
3. For large n (>15): Use gen6x's precomputed lookup tables with gen1a's explicit fail checking
4. Use bit.bit_length() - 1 from gen6x for fast bit extraction in bitwise solvers
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using optimized hybrid dispatch.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Hybrid dispatch: optimal thresholds based on parent performance
    if n <= 10:
        return _solve_simple_bitwise(n)
    elif n <= 15:
        return _solve_bitwise_mrv(n)
    else:
        return _solve_mrv_precomputed(n)


def _solve_simple_bitwise(n: int) -> list[int] | None:
    """Simple row-by-row backtracking with bitmask conflict tracking.

    From gen6x: Uses bit.bit_length() - 1 for faster bit position calculation.
    Best for small n where overhead dominates.
    """
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


def _solve_bitwise_mrv(n: int) -> list[int] | None:
    """Bitwise solver with MRV row selection.

    New hybrid: combines bitwise conflict tracking efficiency with MRV pruning.
    For medium n (11-15) where both matter.
    """
    solution = [-1] * n
    all_cols = (1 << n) - 1

    cols_mask = 0
    diag1_masks = [0] * n  # diag1 mask at each row
    diag2_masks = [0] * n  # diag2 mask at each row
    row_done = [False] * n

    def backtrack(placed: int, cols: int) -> bool:
        if placed == n:
            return True

        # MRV: find row with fewest available positions
        min_count = n + 1
        min_row = -1
        min_available = 0

        for row in range(n):
            if row_done[row]:
                continue

            # Compute available positions for this row using bitmasks
            d1 = diag1_masks[row]
            d2 = diag2_masks[row]
            available = all_cols & ~(cols | d1 | d2)
            count = bin(available).count('1')

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

            # Update diag masks for all unplaced rows
            old_d1 = []
            old_d2 = []
            for r in range(n):
                if not row_done[r]:
                    old_d1.append((r, diag1_masks[r]))
                    old_d2.append((r, diag2_masks[r]))
                    diff = r - row
                    if diff > 0:
                        diag1_masks[r] |= bit << diff
                        diag2_masks[r] |= bit >> diff if diff <= col else 0
                    else:
                        diff = -diff
                        diag1_masks[r] |= bit >> diff if diff <= col else 0
                        diag2_masks[r] |= bit << diff

            if backtrack(placed + 1, cols | bit):
                return True

            # Restore
            for r, v in old_d1:
                diag1_masks[r] = v
            for r, v in old_d2:
                diag2_masks[r] = v

        solution[row] = -1
        row_done[row] = False
        return False

    if backtrack(0, 0):
        return solution
    return None


def _solve_mrv_precomputed(n: int) -> list[int] | None:
    """
    Inlined MRV with precomputed diagonal lookup tables.

    Combines:
    - gen6x: Precomputed diagonal lookup tables (d1_table, d2_table)
    - gen1a: Explicit min_count == 0 fail check for clarity
    - gen8b: Efficient inline base_d1 pattern
    """
    solution = [-1] * n

    # From gen6x: Precompute diagonal lookup tables
    n_minus_1 = n - 1
    d1_table = [[row - col + n_minus_1 for col in range(n)] for row in range(n)]
    d2_table = [[row + col for col in range(n)] for row in range(n)]

    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # Inlined MRV selection with early termination
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            d1_row = d1_table[row]
            d2_row = d2_table[row]
            available = []

            for col in range(n):
                if not col_used[col] and not diag1_used[d1_row[col]] and not diag2_used[d2_row[col]]:
                    available.append(col)

            count = len(available)

            # From gen1a: Immediate fail when row has no options
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
        d1_row = d1_table[row]
        d2_row = d2_table[row]

        for col in min_available:
            d1 = d1_row[col]
            d2 = d2_row[col]

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
