# Generation 0 v3: Each row's shape segment is reflected across the marker column
# The reflected segment has the same length as the original segment

def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find markers (2) and shape color
    markers = {}  # row -> col
    shape_rows = {}  # row -> list of cols
    color = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                markers[r] = c
            elif grid[r][c] != 0:
                if r not in shape_rows:
                    shape_rows[r] = []
                shape_rows[r].append(c)
                color = grid[r][c]

    # Create output with green background
    result = [[3] * cols for _ in range(rows)]

    # Copy original shape
    for r, cs in shape_rows.items():
        for c in cs:
            result[r][c] = color

    # Markers become shape color
    for r, c in markers.items():
        result[r][c] = color

    # For each row with both marker and shape, reflect the shape
    for r in shape_rows:
        if r in markers:
            mc = markers[r]
            shape_cols = shape_rows[r]
            extent = len(shape_cols)
            # Reflect: marker + (extent - 1) cells beyond
            for i in range(1, extent):
                new_c = mc + i
                if 0 <= new_c < cols:
                    result[r][new_c] = color

    # Handle case where marker is on opposite side (shape to right of marker)
    # Need to check direction...

    return result
