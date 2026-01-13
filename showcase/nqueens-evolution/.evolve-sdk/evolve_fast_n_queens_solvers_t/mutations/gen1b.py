"""
N-Queens Solver - Variant gen1b: Iterative Bitwise with Explicit Stack

Approach: Converts recursive backtracking to iterative with explicit stack.
- Eliminates Python function call overhead
- Uses explicit state stack instead of call stack
- Better cache locality for state variables
- Same bitwise conflict detection logic
"""

def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using iterative bitwise backtracking.

    Uses explicit stack to avoid recursion overhead:
    - stack stores (available, cols, diag1, diag2) per row
    - Iterates through positions without function calls
    """
    if n == 0:
        return []

    solution = [-1] * n
    all_ones = (1 << n) - 1

    # Stack: (available_bits, cols, diag1, diag2)
    # Pre-allocate for better performance
    stack_available = [0] * n
    stack_cols = [0] * n
    stack_diag1 = [0] * n
    stack_diag2 = [0] * n

    # Initialize row 0
    row = 0
    cols = 0
    diag1 = 0
    diag2 = 0
    available = all_ones  # All columns available at start

    while row >= 0:
        if row == n:
            # Found a solution
            return solution

        # Compute available if we just arrived at this row
        if solution[row] == -1:
            available = all_ones & ~(cols | diag1 | diag2)
        else:
            # We backtracked here, restore state from stack
            available = stack_available[row]
            cols = stack_cols[row]
            diag1 = stack_diag1[row]
            diag2 = stack_diag2[row]

        if available:
            # Get rightmost set bit
            col_bit = available & -available
            col = col_bit.bit_length() - 1

            solution[row] = col

            # Save remaining available positions for backtracking
            stack_available[row] = available & (available - 1)
            stack_cols[row] = cols
            stack_diag1[row] = diag1
            stack_diag2[row] = diag2

            # Move to next row with updated constraints
            cols |= col_bit
            diag1 = (diag1 | col_bit) << 1
            diag2 = (diag2 | col_bit) >> 1
            row += 1

            # Signal that we're entering a new row
            if row < n:
                solution[row] = -1
        else:
            # No positions available, backtrack
            solution[row] = -1
            row -= 1

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
