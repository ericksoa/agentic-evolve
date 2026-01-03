# Task 1fad071e

## Pattern
Count 2x2 blocks of blue (1) cells and output that count as ones followed by zeros to fill 5 positions.

## Algorithm
Scan the 9x9 grid for 2x2 blocks consisting entirely of 1s. For each position (i,j), check if the 2x2 block starting there equals [1,1,1,1] by concatenating both rows. Count matches and output [1]*count + [0]*(5-count).

## Key Tricks
- `g[i][j:j+2]+g[i+1][j:j+2]==[1]*4` - slice+concat to check 2x2 block (saves vs 4-way chained equality)
- `R=range(8)` - alias saves 2 bytes when range(8) used twice
- `[1]*c+[0]*(5-c)` - direct list construction for output

## Byte History
| Version | Bytes | Change |
|---------|-------|--------|
| v1 | 109 | Initial golfed solution using slice concatenation |
