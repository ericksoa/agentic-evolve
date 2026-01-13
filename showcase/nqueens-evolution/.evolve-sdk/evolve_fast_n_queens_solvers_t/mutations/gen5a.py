"""
N-Queens Solver - Gen5a: Precomputed Diagonal Lookup Tables

Mutation: Precompute all diagonal indices for each (row, col) pair into
lookup tables to eliminate arithmetic operations in the hot inner loop.
- d1_table[row][col] = row - col + n - 1
- d2_table[row][col] = row + col
This trades memory for speed by removing repeated additions/subtractions.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using precomputed diagonal lookup tables.
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

    # Track conflicts using arrays
    col_used = [False] * n
    diag1_used = [False] * (2 * n - 1)
    diag2_used = [False] * (2 * n - 1)

    row_done = [False] * n

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

            for col in range(n):
                if not col_used[col] and not diag1_used[d1_row[col]] and not diag2_used[d2_row[col]]:
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
