def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    if not eights:
        return result

    min_r = min(r for r, c in eights)
    max_r = max(r for r, c in eights)
    min_c = min(c for r, c in eights)
    max_c = max(c for r, c in eights)

    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == 1:
                adj = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] == 8:
                                adj = True
                if adj:
                    result[r][c] = 3

    return result
