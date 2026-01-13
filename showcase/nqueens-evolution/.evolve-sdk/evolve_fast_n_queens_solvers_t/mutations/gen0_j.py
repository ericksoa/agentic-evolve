"""
Variant J: Hybrid Approach - O(n) Construction with Backtrack Fallback
Attempts fast construction first, validates, falls back to optimized backtracking.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Hybrid solver: try O(n) construction, fallback to backtracking."""
    if n == 1:
        return [0]
    if n <= 3:
        return None

    # Try construction first
    board = _try_construction(n)
    if board and _validate(board, n):
        return board

    # Fallback to optimized backtracking
    return _optimized_backtrack(n)


def _try_construction(n: int) -> list[int]:
    """Try O(n) construction."""
    board = [0] * n

    # Algorithm: evens on first half, odds on second half with specific rules
    if n % 6 == 2:
        j = 0
        for i in range(2, n + 1, 2):
            board[j] = i - 1 if i != 2 else n - 1
            j += 1
        board[n // 2 - 1] = 1
        for i in range(1, n, 2):
            board[j] = i - 1 if i != 1 else 0
            j += 1
    elif n % 6 == 3:
        j = 0
        for i in range(4, n + 1, 2):
            board[j] = i - 1
            j += 1
        board[j] = 1
        j += 1
        board[j] = 3
        j += 1
        board[j] = n - 1
        j += 1
        for i in range(5, n, 2):
            board[j] = i - 1
            j += 1
        board[j] = 0
        j += 1
        board[j] = 2
    else:
        # Standard: odds then evens (0-indexed columns)
        j = 0
        for i in range(1, n, 2):
            board[j] = i
            j += 1
        for i in range(0, n, 2):
            board[j] = i
            j += 1

    return board


def _validate(board: list[int], n: int) -> bool:
    """Validate solution."""
    if len(set(board)) != n:
        return False
    for i in range(n):
        if board[i] < 0 or board[i] >= n:
            return False
        for j in range(i):
            if abs(board[i] - board[j]) == i - j:
                return False
    return True


def _optimized_backtrack(n: int) -> list[int]:
    """Optimized backtracking with bit operations."""
    board = [-1] * n
    all_cols = (1 << n) - 1

    def solve(row: int, cols: int, d1: int, d2: int) -> bool:
        if row == n:
            return True

        available = all_cols & ~(cols | d1 | d2)
        while available:
            pos = available & -available
            available ^= pos
            col = pos.bit_length() - 1
            board[row] = col

            if solve(row + 1, cols | pos, (d1 | pos) << 1, (d2 | pos) >> 1):
                return True

        return False

    solve(0, 0, 0, 0)
    return board


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        valid = _validate(result, n) if result else False
        print(f"N={n}: {result} (valid={valid})")
