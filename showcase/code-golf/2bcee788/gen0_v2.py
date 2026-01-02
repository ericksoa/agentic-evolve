# Generation 0 v2: Improved understanding
# The shape is reflected over the marker (2) positions, which act as a mirror axis

def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find markers (2) and shape color
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

    # Create output with green background
    result = [[3] * cols for _ in range(rows)]

    # Copy original shape
    for r, c in shape:
        result[r][c] = color

    # Markers become shape color
    for r, c in markers:
        result[r][c] = color

    # Determine reflection direction by marker positions relative to shape
    # Check if markers are horizontal or vertical line relative to shape
    marker_rows = set(r for r, c in markers)
    marker_cols = set(c for r, c in markers)
    shape_rows = set(r for r, c in shape)
    shape_cols = set(c for r, c in shape)

    # If all markers share same column, it's a vertical line -> horizontal reflection
    # If all markers share same row, it's a horizontal line -> vertical reflection

    if len(marker_cols) == 1:  # Vertical marker line -> reflect horizontally
        mc = list(marker_cols)[0]
        for r, c in shape:
            new_c = 2 * mc - c
            if 0 <= new_c < cols:
                result[r][new_c] = color
    elif len(marker_rows) == 1:  # Horizontal marker line -> reflect vertically
        mr = list(marker_rows)[0]
        for r, c in shape:
            new_r = 2 * mr - r
            if 0 <= new_r < rows:
                result[new_r][c] = color

    return result
