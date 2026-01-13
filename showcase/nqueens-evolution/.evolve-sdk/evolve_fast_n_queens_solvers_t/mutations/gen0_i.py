"""
Variant I: Explicit Construction with Modular Arithmetic
Uses proven O(n) construction algorithm based on modular properties.
"""

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using explicit construction."""
    if n == 1:
        return [0]
    if n <= 3:
        return None

    board = [-1] * n

    # Use explicit construction based on n mod 6
    remainder = n % 6

    if remainder == 2:
        _construct_r2(board, n)
    elif remainder == 3:
        _construct_r3(board, n)
    else:
        _construct_standard(board, n)

    return board


def _construct_standard(board: list[int], n: int) -> None:
    """Standard construction: odds then evens."""
    idx = 0
    # Place queens at odd columns first
    for col in range(1, n, 2):
        board[idx] = col
        idx += 1
    # Then even columns
    for col in range(0, n, 2):
        board[idx] = col
        idx += 1


def _construct_r2(board: list[int], n: int) -> None:
    """Construction for n ≡ 2 (mod 6)."""
    half = n // 2
    # Even indices get: 1, 3, 5, ..., n-1 shifted
    # Odd indices get: 0, 2, 4, ..., n-2 shifted

    idx = 0
    # First half: 1, 3, 5, ... but start from 3
    for col in range(3, n, 2):
        board[idx] = col
        idx += 1
    board[idx] = 1
    idx += 1

    # Second half: 0, 2, 4, ... but end with n-1
    board[idx] = n - 1
    idx += 1
    for col in range(2, n - 1, 2):
        board[idx] = col
        idx += 1
    board[idx] = 0


def _construct_r3(board: list[int], n: int) -> None:
    """Construction for n ≡ 3 (mod 6)."""
    idx = 0

    # Evens first, but shifted: 4, 6, 8, ..., n-1, 2
    for col in range(4, n, 2):
        board[idx] = col
        idx += 1
    board[idx] = 2
    idx += 1
    board[idx] = 0
    idx += 1

    # Then odds shifted: 5, 7, 9, ..., n-2, 1, 3
    for col in range(5, n, 2):
        board[idx] = col
        idx += 1
    board[idx] = 1
    idx += 1
    board[idx] = 3
    idx += 1


if __name__ == "__main__":
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
