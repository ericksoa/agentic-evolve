# Generation 0 v9: Always use marker line as edge, not shape-dependent

def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    markers = set()
    shape = set()
    color = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                markers.add((r, c))
            elif grid[r][c] != 0:
                shape.add((r, c))
                color = grid[r][c]

    marker_rows = set(r for r, c in markers)
    marker_cols = set(c for r, c in markers)
    shape_rows = set(r for r, c in shape)
    shape_cols = set(c for r, c in shape)

    result = [[3] * cols for _ in range(rows)]

    for r, c in shape:
        result[r][c] = color
    for r, c in markers:
        result[r][c] = color

    if len(marker_rows) == 1:
        # Horizontal line of markers
        mr = list(marker_rows)[0]
        shape_below = min(shape_rows) > mr
        edge = mr + 0.5 if shape_below else mr - 0.5

        for r, c in shape:
            new_r = int(2 * edge - r)
            if 0 <= new_r < rows:
                result[new_r][c] = color

    elif len(marker_cols) == 1:
        # Vertical line of markers
        mc = list(marker_cols)[0]
        shape_left = max(shape_cols) < mc
        edge = mc - 0.5 if shape_left else mc + 0.5

        for r, c in shape:
            new_c = int(2 * edge - c)
            if 0 <= new_c < cols:
                result[r][new_c] = color

    return result
