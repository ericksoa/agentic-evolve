# Generation 0 v6: Correct reflection formula

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

    # Determine direction based on marker configuration
    marker_rows = set(r for r, c in markers)
    marker_cols = set(c for r, c in markers)
    shape_rows = set(r for r, c in shape)
    shape_cols = set(c for r, c in shape)

    # Create output with green background
    result = [[3] * cols for _ in range(rows)]

    # Copy original shape
    for r, c in shape:
        result[r][c] = color

    # Markers become shape color
    for r, c in markers:
        result[r][c] = color

    if len(marker_rows) == 1:
        # Horizontal line of markers -> vertical reflection
        mr = list(marker_rows)[0]
        # Determine if shape is above or below
        shape_below = min(shape_rows) > mr

        # For each column, reflect shape cells
        shape_by_col = {}
        for r, c in shape:
            if c not in shape_by_col:
                shape_by_col[c] = []
            shape_by_col[c].append(r)

        for c, rs in shape_by_col.items():
            for r in rs:
                if shape_below:
                    new_r = 2 * mr + 1 - r
                else:
                    new_r = 2 * mr - 1 - r
                if 0 <= new_r < rows:
                    result[new_r][c] = color

    elif len(marker_cols) == 1:
        # Vertical line of markers -> horizontal reflection
        mc = list(marker_cols)[0]
        # Determine if shape is left or right
        shape_left = max(shape_cols) < mc

        # For each row, reflect shape cells
        shape_by_row = {}
        for r, c in shape:
            if r not in shape_by_row:
                shape_by_row[r] = []
            shape_by_row[r].append(c)

        for r, cs in shape_by_row.items():
            for c in cs:
                if shape_left:
                    new_c = 2 * mc + 1 - c
                else:
                    new_c = 2 * mc - 1 - c
                if 0 <= new_c < cols:
                    result[r][new_c] = color

    return result
