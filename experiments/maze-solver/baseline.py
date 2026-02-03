"""Baseline: Manhattan Distance heuristic for A* maze solving."""

def heuristic(row, col, goal_row, goal_col, grid, size):
    """Manhattan distance - always admissible for 4-connected grid."""
    return abs(row - goal_row) + abs(col - goal_col)
