# Generation 0: Initial readable solution
# Understanding the transformation:
# 1. Background 0 -> 3
# 2. Find color 2 (marker) and the other color (shape)
# 3. Reflect the shape across the marker line
# 4. Marker becomes shape color

def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find marker color 2 positions and shape color
    markers = []
    shape_cells = []
    shape_color = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                markers.append((r, c))
            elif grid[r][c] != 0:
                shape_cells.append((r, c))
                shape_color = grid[r][c]

    # Create output with green background
    result = [[3] * cols for _ in range(rows)]

    # Copy original shape
    for r, c in shape_cells:
        result[r][c] = shape_color

    # Find reflection axis based on marker positions relative to shape
    # If markers are to the right of shape, reflect horizontally
    # If markers are below shape, reflect vertically
    # etc.

    # Get bounding box of shape
    min_r = min(r for r, c in shape_cells)
    max_r = max(r for r, c in shape_cells)
    min_c = min(c for r, c in shape_cells)
    max_c = max(c for r, c in shape_cells)

    # Find marker direction
    marker_r = markers[0][0]
    marker_c = markers[0][1]

    # Determine reflection direction
    if marker_c > max_c:  # Markers to the right
        # Reflect horizontally
        for r, c in shape_cells:
            new_c = marker_c + (marker_c - c)
            if 0 <= new_c < cols:
                result[r][new_c] = shape_color
        for mr, mc in markers:
            result[mr][mc] = shape_color
    elif marker_c < min_c:  # Markers to the left
        for r, c in shape_cells:
            new_c = marker_c - (c - marker_c)
            if 0 <= new_c < cols:
                result[r][new_c] = shape_color
        for mr, mc in markers:
            result[mr][mc] = shape_color
    elif marker_r > max_r:  # Markers below
        for r, c in shape_cells:
            new_r = marker_r + (marker_r - r)
            if 0 <= new_r < rows:
                result[new_r][c] = shape_color
        for mr, mc in markers:
            result[mr][mc] = shape_color
    elif marker_r < min_r:  # Markers above
        for r, c in shape_cells:
            new_r = marker_r - (r - marker_r)
            if 0 <= new_r < rows:
                result[new_r][c] = shape_color
        for mr, mc in markers:
            result[mr][mc] = shape_color

    return result
