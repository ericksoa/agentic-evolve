# Task d631b094

## Pattern
Collect all non-zero cells from 3x3 grid into a single-row output.

## Algorithm
Flatten the grid and filter out zeros. All non-zero values have the same color in each example, so the output is just a 1D list of all colored cells.

## Key Tricks
- `for r in g for c in r` - nested iteration to flatten grid
- `if c` - filter zeros using truthiness
- Double brackets `[[...]]` for required output format

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 47 | Initial lambda solution |
