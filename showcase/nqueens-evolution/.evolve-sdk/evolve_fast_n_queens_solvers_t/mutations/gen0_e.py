"""
Variant E: Min-Conflicts Heuristic with Random Restarts
Local search approach - starts with random placement, fixes conflicts.
"""

import random

def solve_nqueens(n: int) -> list[int] | None:
    """Solve N-Queens using min-conflicts with random restarts."""
    max_restarts = 100
    max_iterations = n * 5

    def count_conflicts(board: list[int], row: int, col: int) -> int:
        """Count conflicts if queen is placed at (row, col)."""
        conflicts = 0
        for r in range(n):
            if r == row:
                continue
            c = board[r]
            if c == col or abs(c - col) == abs(r - row):
                conflicts += 1
        return conflicts

    def total_conflicts(board: list[int]) -> int:
        """Count total conflicts in board."""
        total = 0
        for row in range(n):
            total += count_conflicts(board, row, board[row])
        return total // 2  # Each conflict counted twice

    def find_conflicting_row(board: list[int]) -> int:
        """Find a row with conflicts."""
        conflicting = []
        for row in range(n):
            if count_conflicts(board, row, board[row]) > 0:
                conflicting.append(row)
        return random.choice(conflicting) if conflicting else -1

    for _ in range(max_restarts):
        # Random initial placement - one queen per row
        board = list(range(n))
        random.shuffle(board)

        for _ in range(max_iterations):
            if total_conflicts(board) == 0:
                return board

            # Find a conflicting queen
            row = find_conflicting_row(board)
            if row == -1:
                return board

            # Find column with minimum conflicts
            min_conflicts = n + 1
            best_cols = []
            for col in range(n):
                c = count_conflicts(board, row, col)
                if c < min_conflicts:
                    min_conflicts = c
                    best_cols = [col]
                elif c == min_conflicts:
                    best_cols.append(col)

            board[row] = random.choice(best_cols)

    # Fallback to simple backtracking
    return _backtrack_solve(n)


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
    random.seed(42)
    for n in [8, 12, 16, 20]:
        result = solve_nqueens(n)
        print(f"N={n}: {result}")
