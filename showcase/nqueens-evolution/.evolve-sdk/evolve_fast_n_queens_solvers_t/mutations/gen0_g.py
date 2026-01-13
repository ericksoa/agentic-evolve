"""
Variant G: Known Solution Patterns with Fallback
Uses mathematical patterns for certain N, falls back to search.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using known patterns where possible."""
    if n == 1:
        return [0]
    if n <= 3:
        return None  # No solution for 2, 3

    # Use known construction for efficient solutions
    if n % 6 == 2:
        return _pattern_mod6_2(n)
    elif n % 6 == 3:
        return _pattern_mod6_3(n)
    else:
        return _pattern_general(n)


def _pattern_mod6_2(n: int) -> list[int]:
    """Pattern for n ≡ 2 (mod 6)."""
    # Place queens: 2,4,6,...,n, 3,1 arrangement
    result = [-1] * n
    k = 0
    for i in range(1, n, 2):
        result[k] = i
        k += 1
    for i in range(0, n, 2):
        result[k] = i
        k += 1

    # Adjust for mod 6 = 2 case
    # Swap positions to fix diagonals
    half = n // 2
    if n % 6 == 2:
        # Swap 1st and 3rd in odd positions, 0th and 2nd in even
        result[0], result[2] = result[2], result[0]
        result[half], result[half + 2] = result[half + 2], result[half]

    return result


def _pattern_mod6_3(n: int) -> list[int]:
    """Pattern for n ≡ 3 (mod 6)."""
    result = [-1] * n
    k = 0
    for i in range(1, n, 2):
        result[k] = i
        k += 1
    for i in range(0, n, 2):
        result[k] = i
        k += 1

    # Adjustments for mod 6 = 3
    half = n // 2
    result[0], result[2] = result[2], result[0]
    result[half], result[half + 2] = result[half + 2], result[half]
    result[1], result[3] = result[3], result[1]

    return result


def _pattern_general(n: int) -> list[int]:
    """General pattern: evens then odds or shifted."""
    result = [-1] * n

    # Construct solution: evens first, then odds
    k = 0
    for i in range(1, n, 2):  # Odd columns: 1, 3, 5, ...
        result[k] = i
        k += 1
    for i in range(0, n, 2):  # Even columns: 0, 2, 4, ...
        result[k] = i
        k += 1

    # Verify and fix if needed
    if _is_valid(result):
        return result

    # Fallback to backtracking
    return _backtrack_solve(n)


def _is_valid(board: list[int]) -> bool:
    """Check if board is valid."""
    n = len(board)
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j]:
                return False
            if abs(board[i] - board[j]) == j - i:
                return False
    return True


def _backtrack_solve(n: int) -> list[int] | None:
    """Fallback backtracking solver."""
    board = [-1] * n
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)
    diag2 = [False] * (2 * n - 1)

    def backtrack(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            d1 = row - col + n - 1
            d2 = row + col
            if not cols[col] and not diag1[d1] and not diag2[d2]:
                board[row] = col
                cols[col] = diag1[d1] = diag2[d2] = True
                if backtrack(row + 1):
                    return True
                cols[col] = diag1[d1] = diag2[d2] = False
        return False

    backtrack(0)
    return board


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
