"""
N-Queens Solver - Variant B: Iterative with Explicit Stack

Approach: Converts recursion to iteration using explicit stack.
- Avoids function call overhead
- Uses sets for O(1) conflict checking
- More control over execution flow
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using iterative backtracking with explicit stack.

    Each stack entry contains (row, col_iterator, state_snapshot)
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    row = 0
    col = 0

    while True:
        # Find next valid column in current row
        placed = False
        while col < n:
            d1 = row - col
            d2 = row + col

            if col not in cols and d1 not in diag1 and d2 not in diag2:
                # Place queen
                solution[row] = col
                cols.add(col)
                diag1.add(d1)
                diag2.add(d2)
                placed = True
                break
            col += 1

        if placed:
            if row == n - 1:
                # Found a solution
                return solution
            # Move to next row
            row += 1
            col = 0
        else:
            # Backtrack
            if row == 0:
                # No solution exists
                return None

            row -= 1
            col = solution[row]

            # Remove the queen from this position
            d1 = row - col
            d2 = row + col
            cols.discard(col)
            diag1.discard(d1)
            diag2.discard(d2)

            # Try next column
            col += 1


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
