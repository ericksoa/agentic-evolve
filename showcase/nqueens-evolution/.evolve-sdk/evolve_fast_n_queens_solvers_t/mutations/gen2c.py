"""
N-Queens Solver - Gen2c: Bitwise Backtracking with De Bruijn Bit-Position Lookup

Mutation: Replace bit_length() with De Bruijn multiplication lookup table.
- Uses De Bruijn sequence for O(1) bit-to-index conversion
- Eliminates function call overhead of bit_length()
- Single table lookup vs Python method dispatch
"""

# De Bruijn sequence for 64-bit integers - maps isolated bit to position
# This is a classic optimization for finding bit position of isolated bits
_DEBRUIJN64 = 0x03f79d71b4cb0a89
_DEBRUIJN_TABLE = [0] * 64
for _i in range(64):
    _DEBRUIJN_TABLE[(_DEBRUIJN64 << _i >> 58) & 63] = _i


def solve_nqueens(n: int) -> list[int] | None:
    """
    Solve N-Queens using bitwise backtracking with De Bruijn bit lookup.

    Uses bitmasks to track:
    - cols: occupied columns
    - diag1: occupied left-down diagonals (row - col)
    - diag2: occupied right-down diagonals (row + col)
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    solution = [-1] * n
    all_cols = (1 << n) - 1

    # Local reference to lookup table for faster access
    debruijn = _DEBRUIJN64
    table = _DEBRUIJN_TABLE

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> bool:
        if row == n:
            return True

        # Available positions: not in any conflict set
        available = all_cols & ~(cols | diag1 | diag2)

        while available:
            # Get rightmost set bit (lowest column available)
            col_bit = available & -available

            # De Bruijn bit position lookup - faster than bit_length()
            col = table[(col_bit * debruijn >> 58) & 63]

            solution[row] = col

            # Try placing queen here
            if backtrack(
                row + 1,
                cols | col_bit,
                (diag1 | col_bit) << 1,
                (diag2 | col_bit) >> 1
            ):
                return True

            # Remove this bit from available
            available &= available - 1

        return False

    if backtrack(0, 0, 0, 0):
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
