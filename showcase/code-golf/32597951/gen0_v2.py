def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find the repeating tile period by finding rows/cols without 8s
    # and checking for periodicity

    # First, find rows without 8s
    safe_rows = [r for r in range(rows) if 8 not in grid[r]]
    safe_cols = [c for c in range(cols) if all(grid[r][c] != 8 for r in range(rows))]

    if not safe_rows or not safe_cols:
        return result

    # Find period by checking when pattern repeats
    def find_period(indices, get_vals):
        for p in range(1, len(indices)):
            ok = True
            for i in range(len(indices) - p):
                if get_vals(indices[i]) != get_vals(indices[i + p]):
                    ok = False
                    break
            if ok:
                return p
        return len(indices)

    # Now for each cell with 8, determine what it should be
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                # Find a reference cell that tells us what this should be
                # Look at same column in a safe row, same row in safe col
                found_val = None

                # Try to find periodicity
                for sr in safe_rows:
                    for sc in safe_cols:
                        # Check if grid[sr][c] == grid[sr][sc] when (c-sc) matches period
                        pass

                # Simple approach: find any row without 8 in this column
                for ref_r in range(rows):
                    if grid[ref_r][c] != 8:
                        # Check if ref_r and r have same pattern position
                        # by checking a safe column
                        for sc in safe_cols:
                            if grid[r][sc] == grid[ref_r][sc]:
                                found_val = grid[ref_r][c]
                                break
                        if found_val is not None:
                            break

                if found_val == 1:
                    result[r][c] = 3

    return result
