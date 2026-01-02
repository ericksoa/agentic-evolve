# Generation 0 v4: Handle both horizontal and vertical reflections

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

    # Determine direction: check if markers form horizontal or vertical line
    marker_rows = set(r for r, c in markers)
    marker_cols = set(c for r, c in markers)

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
        # For each column, find shape extent and reflect
        for c in marker_cols:
            # Find shape cells in this column
            shape_in_col = sorted([r for r, cc in shape if cc == c])
            if shape_in_col:
                extent = len(shape_in_col)
                # Shape is below marker, reflect upward
                if min(shape_in_col) > mr:
                    for i in range(1, extent):
                        new_r = mr - i
                        if 0 <= new_r < rows:
                            result[new_r][c] = color
                # Shape is above marker, reflect downward
                else:
                    for i in range(1, extent):
                        new_r = mr + i
                        if 0 <= new_r < rows:
                            result[new_r][c] = color

    elif len(marker_cols) == 1:
        # Vertical line of markers -> horizontal reflection
        mc = list(marker_cols)[0]
        # For each row with marker, find shape extent and reflect
        for r in marker_rows:
            # Find shape cells in this row
            shape_in_row = sorted([c for rr, c in shape if rr == r])
            if shape_in_row:
                extent = len(shape_in_row)
                # Shape is left of marker, reflect right
                if max(shape_in_row) < mc:
                    for i in range(1, extent):
                        new_c = mc + i
                        if 0 <= new_c < cols:
                            result[r][new_c] = color
                # Shape is right of marker, reflect left
                else:
                    for i in range(1, extent):
                        new_c = mc - i
                        if 0 <= new_c < cols:
                            result[r][new_c] = color

    return result
