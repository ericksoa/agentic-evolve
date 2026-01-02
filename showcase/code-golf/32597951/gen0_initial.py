def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find all cells with 8 to get the marked region
    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]

    if not eights:
        return result

    # Find bounding box of 8s
    min_r = min(r for r, c in eights)
    max_r = max(r for r, c in eights)
    min_c = min(c for r, c in eights)
    max_c = max(c for r, c in eights)

    # For cells inside the region marked by 8s, if original was 1 and now is 8,
    # we need to figure out what the pattern should be and mark differences as 3

    # The pattern repeats - find repeating period
    # Try to find what value should be at each position by looking at non-8 cells

    for r, c in eights:
        # Look for pattern outside the 8 region
        # Find what value should be at this position based on periodicity

        # Check if this cell should be 1 based on the pattern
        # We look at equivalent positions outside the 8 region

        # Try various periods to find the repeating pattern
        for period_r in range(1, rows):
            for period_c in range(1, cols):
                # Check multiple offsets
                found = False
                for dr in [-period_r, period_r]:
                    nr = r + dr
                    if 0 <= nr < rows:
                        if grid[nr][c] != 8:
                            if grid[nr][c] == 1:
                                result[r][c] = 3
                            found = True
                            break
                if found:
                    break
            if found:
                break

    return result
