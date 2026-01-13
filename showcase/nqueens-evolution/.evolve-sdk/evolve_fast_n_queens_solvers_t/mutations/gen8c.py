"""
N-Queens Solver - Gen8c: Branch Elimination via Integer Arithmetic

Parent: gen5a (fitness 18120.44)
Mutation: Replace boolean arrays with integer arrays (0/1) and use arithmetic
instead of logical operations to reduce branch mispredictions.

Key change: Instead of:
    if not col_used[col] and not diag1_used[d1] and not diag2_used[d2]:
Use:
    if col_used[col] + diag1_used[d1] + diag2_used[d2] == 0:

This replaces 3 conditional checks with arithmetic addition and one comparison,
which can be faster due to fewer branch mispredictions in the CPU pipeline.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using precomputed diagonal lookup tables with branch elimination.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n

    # Precompute diagonal lookup tables
    n_minus_1 = n - 1
    d1_table = [[row - col + n_minus_1 for col in range(n)] for row in range(n)]
    d2_table = [[row + col for col in range(n)] for row in range(n)]

    # Track conflicts using integer arrays (0=free, 1=used)
    # This enables arithmetic-based branch elimination
    col_used = [0] * n
    diag1_used = [0] * (2 * n - 1)
    diag2_used = [0] * (2 * n - 1)

    row_done = [0] * n

    def backtrack(placed: int) -> bool:
        if placed == n:
            return True

        # MRV: select most constrained row
        min_count = n + 1
        min_row = -1
        min_available = None

        for row in range(n):
            if row_done[row]:
                continue

            # Use lookup tables instead of computing indices
            d1_row = d1_table[row]
            d2_row = d2_table[row]
            available = []

            # Branch elimination: use arithmetic instead of multiple logical operations
            for col in range(n):
                # Single comparison vs three boolean checks
                if col_used[col] + diag1_used[d1_row[col]] + diag2_used[d2_row[col]] == 0:
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
        row_done[row] = 1
        d1_row = d1_table[row]
        d2_row = d2_table[row]

        for col in min_available:
            d1 = d1_row[col]
            d2 = d2_row[col]

            solution[row] = col
            col_used[col] = 1
            diag1_used[d1] = 1
            diag2_used[d2] = 1

            if backtrack(placed + 1):
                return True

            col_used[col] = 0
            diag1_used[d1] = 0
            diag2_used[d2] = 0

        solution[row] = -1
        row_done[row] = 0
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
