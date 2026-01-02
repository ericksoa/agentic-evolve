# Generation 0 v8: Use marker position for edge in horizontal markers case

def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find markers (2), shape cells, and shape color
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

        # Group shape by column
        shape_by_col = {}
        for r, c in shape:
            if c not in shape_by_col:
                shape_by_col[c] = []
            shape_by_col[c].append(r)

        for c, rs in shape_by_col.items():
            # Edge is at marker position
            if shape_below:
                edge = mr + 0.5
            else:
                edge = mr - 0.5
            for r in rs:
                new_r = int(2 * edge - r)
                if 0 <= new_r < rows:
                    result[new_r][c] = color

    elif len(marker_cols) == 1:
        # Vertical line of markers
        mc = list(marker_cols)[0]
        shape_left = max(shape_cols) < mc

        # Group shape by row
        shape_by_row = {}
        for r, c in shape:
            if r not in shape_by_row:
                shape_by_row[r] = []
            shape_by_row[r].append(c)

        for r, cs in shape_by_row.items():
            # Edge depends on shape position in this row
            if shape_left:
                edge = max(cs) + 0.5
            else:
                edge = min(cs) - 0.5
            for c in cs:
                new_c = int(2 * edge - c)
                if 0 <= new_c < cols:
                    result[r][new_c] = color

    return result
