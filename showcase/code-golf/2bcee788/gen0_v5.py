# Generation 0 v5: Reflect ALL shape cells across the marker line

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

    # Determine direction
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
        # Horizontal line of markers at row mr -> vertical reflection
        mr = list(marker_rows)[0]
        # For each column, find shape cells and reflect
        shape_by_col = {}
        for r, c in shape:
            if c not in shape_by_col:
                shape_by_col[c] = []
            shape_by_col[c].append(r)

        for c, shape_rows_in_col in shape_by_col.items():
            shape_rows_in_col.sort()
            extent = len(shape_rows_in_col)
            # Shape is below marker, reflect upward
            if min(shape_rows_in_col) > mr:
                for i in range(extent):
                    new_r = mr - (min(shape_rows_in_col) - mr) + i
                    # Actually simpler: reflect each cell
                for sr in shape_rows_in_col:
                    new_r = 2 * mr - sr
                    if 0 <= new_r < rows:
                        result[new_r][c] = color
            # Shape is above marker, reflect downward
            elif max(shape_rows_in_col) < mr:
                for sr in shape_rows_in_col:
                    new_r = 2 * mr - sr
                    if 0 <= new_r < rows:
                        result[new_r][c] = color

    elif len(marker_cols) == 1:
        # Vertical line of markers at col mc -> horizontal reflection
        mc = list(marker_cols)[0]
        # For each row, find shape cells and reflect
        shape_by_row = {}
        for r, c in shape:
            if r not in shape_by_row:
                shape_by_row[r] = []
            shape_by_row[r].append(c)

        for r, shape_cols_in_row in shape_by_row.items():
            shape_cols_in_row.sort()
            extent = len(shape_cols_in_row)
            # Shape is left of marker, reflect right
            if max(shape_cols_in_row) < mc:
                for sc in shape_cols_in_row:
                    new_c = 2 * mc - sc
                    if 0 <= new_c < cols:
                        result[r][new_c] = color
            # Shape is right of marker, reflect left
            elif min(shape_cols_in_row) > mc:
                for sc in shape_cols_in_row:
                    new_c = 2 * mc - sc
                    if 0 <= new_c < cols:
                        result[r][new_c] = color

    return result
