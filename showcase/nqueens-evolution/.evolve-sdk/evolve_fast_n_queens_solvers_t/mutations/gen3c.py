"""
N-Queens Solver - Gen3c: MRV with Precomputed Combined Attack Bits

Mutation: Lookup table optimization.
- Precompute d1_bit AND d2_bit for each (row, col) in a single lookup
- Use tuples instead of lists for constant-time indexed access
- Store diagonal bits directly (not indices) to avoid 1<<x at runtime
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using MRV heuristic with precomputed attack bits.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n

    # Precompute ALL bit patterns as tuples for fast lookup
    col_bits = tuple(1 << col for col in range(n))

    # Precompute diagonal bits directly (row-indexed tuples of col-indexed tuples)
    # This avoids 1 << computation at runtime
    d1_bits = tuple(tuple(1 << (row - col + n - 1) for col in range(n)) for row in range(n))
    d2_bits = tuple(tuple(1 << (row + col) for col in range(n)) for row in range(n))

    # Bitmasks for tracking conflicts
    col_mask = 0
    diag1_mask = 0
    diag2_mask = 0

    row_done = [False] * n

    def backtrack(placed: int) -> bool:
        nonlocal col_mask, diag1_mask, diag2_mask

        if placed == n:
            return True

        # MRV: select most constrained row
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            # Build available list - use precomputed bits
            row_d1 = d1_bits[row]
            row_d2 = d2_bits[row]
            available = []
            for col in range(n):
                if not (col_mask & col_bits[col] or diag1_mask & row_d1[col] or diag2_mask & row_d2[col]):
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
