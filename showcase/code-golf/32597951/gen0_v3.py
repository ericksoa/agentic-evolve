def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find bounding box of 8s
    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    if not eights:
        return result

    min_r = min(r for r, c in eights)
    max_r = max(r for r, c in eights)
    min_c = min(c for r, c in eights)
    max_c = max(c for r, c in eights)

    # Find row period using rows outside the 8 region
    def find_row_period():
        safe_rows = [r for r in range(rows) if all(grid[r][c] != 8 for c in range(cols))]
        if len(safe_rows) < 2:
            return 1
        for p in range(1, len(safe_rows)):
            ok = True
            for i in range(len(safe_rows) - p):
                if grid[safe_rows[i]] != grid[safe_rows[i + p]]:
                    ok = False
                    break
            if ok:
                return p
        return len(safe_rows)

    # Find col period using cols outside the 8 region
    def find_col_period():
        safe_cols = [c for c in range(cols) if all(grid[r][c] != 8 for r in range(rows))]
        if len(safe_cols) < 2:
            return 1
        for p in range(1, len(safe_cols)):
            ok = True
            for i in range(len(safe_cols) - p):
                col_i = [grid[r][safe_cols[i]] for r in range(rows)]
                col_ip = [grid[r][safe_cols[i + p]] for r in range(rows)]
                if col_i != col_ip:
                    ok = False
                    break
            if ok:
                return p
        return len(safe_cols)

    rp = find_row_period()
    cp = find_col_period()

    # Find reference rows and cols (outside 8 region)
    safe_rows = [r for r in range(rows) if all(grid[r][c] != 8 for c in range(cols))]
    safe_cols = [c for c in range(cols) if all(grid[r][c] != 8 for r in range(rows))]

    # For each cell in the 8 region that has value 1
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == 1:
                # Find expected value
                ref_r = safe_rows[r % rp] if safe_rows else r
                ref_c = safe_cols[c % cp] if safe_cols else c
                expected = grid[ref_r][ref_c]
                if expected == 0:
                    result[r][c] = 3

    return result
