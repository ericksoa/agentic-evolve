# ARC Task 27a28665

## Pattern: Shape Classification (Easy)

This task classifies 3x3 binary patterns into one of four categories based on the spatial arrangement of non-zero cells.

## Task Analysis

Input: 3x3 grid with non-zero values forming a specific pattern
Output: Single-cell grid [[n]] where n indicates the pattern type:
- **1**: Upper-left corner pattern (7-shape)
- **2**: X/diagonal pattern (corners + center)
- **3**: Upper-right corner pattern
- **6**: Plus/cross pattern (edges + center)

## Evolution Log

### Generation 1: Initial Solution (228 bytes)
```python
def solve(g):
    # Convert to binary mask and then to integer
    n = sum((1 if g[i//3][i%3] else 0) * (2**i) for i in range(9))
    # Lookup table: 171->1, 341->2, 118->3, 186->6
    return [[{171:1, 341:2, 118:3, 186:6}[n]]]
```
Approach: Encode 3x3 grid as 9-bit integer, use dictionary lookup.

### Generation 2: Whitespace Removal (110 bytes)
```python
def solve(g):
 n=sum((1if g[i//3][i%3]else 0)*2**i for i in range(9))
 return[[{171:1,341:2,118:3,186:6}[n]]]
```
Applied: Single-space indent, removed spaces around operators.

### Generation 3: Boolean Coercion (102 bytes)
```python
def solve(g):
 n=sum((g[i//3][i%3]>0)*2**i for i in range(9))
 return[[{171:1,341:2,118:3,186:6}[n]]]
```
Applied: Replace `1 if x else 0` with `x>0` (evaluates to True/False, multiplies correctly).

### Generation 4: Bit Shift (102 bytes)
```python
def solve(g):
 n=sum((g[i//3][i%3]>0)<<i for i in range(9))
 return[[{171:1,341:2,118:3,186:6}[n]]]
```
Applied: Replace `*2**i` with `<<i` (same length but cleaner).

### Generation 5: Modulo Hash Discovery (90 bytes)
```python
def solve(g):
 n=sum((g[i//3][i%3]>0)<<i for i in range(9))
 return[[[6,0,0,1,3,2][n%6]]]
```
Breakthrough: Discovered that `n%6` uniquely maps all patterns to indices 0-5, allowing list lookup instead of dictionary.

### Generation 6: Modulo 13 Optimization (89 bytes)
```python
def solve(g):
 n=sum((g[i//3][i%3]>0)<<i for i in range(9))
 return[[[0,3,1,2,6][n%13]]]
```
Applied: Found mod 13 gives smaller max index (4), allowing 5-element lookup.

### Generation 7: Lambda Conversion (78 bytes)
```python
solve=lambda g:[[[0,3,1,2,6][sum((g[i//3][i%3]>0)<<i for i in range(9))%13]]]
```
Applied: Convert to lambda, inline the sum.

### Generation 8: Conditional Filter (77 bytes)
```python
solve=lambda g:[[(0,3,1,2,6)[sum(1<<i for i in range(9)if g[i//3][i%3])%13]]]
```
Applied: Use `if` filter instead of `>0` multiplication, tuple instead of list for lookup.

### Generation 9: Bit-Packed Magic Number (75 bytes)
```python
solve=lambda g:[[401712>>sum(4<<i for i in range(9)if g[i//3][i%3])%52&15]]
```
Breakthrough: Pack lookup values into single integer (0x62130 = 401712), extract with bit shifts.
The formula `(4*hash)%52` maps to shift positions 4, 8, 12, 16 for values 3, 1, 2, 6.

### Generation 10: Optimal Magic Number (70 bytes)
```python
solve=lambda g:[[393>>sum(7<<i for i in range(9)if g[i//3][i%3])%9&7]]
```
Breakthrough: Found optimal parameters through exhaustive search:
- Magic number: 393 (encodes values at bit positions 0, 2, 6, 7)
- Multiplier: 7 (maps hash values to correct shift positions)
- Modulo: 9 (gives unique indices)
- Mask: 7 (3-bit extraction, since max value is 6)

## Champion Solution

```python
solve=lambda g:[[393>>sum(7<<i for i in range(9)if g[i//3][i%3])%9&7]]
```

**70 bytes** | Score: 2430 | Fitness: 0.972

### How It Works

1. **Hash Computation**: `sum(7<<i for i in range(9)if g[i//3][i%3])`
   - Iterates through 9 cell positions
   - For each non-zero cell at position i, adds `7 * 2^i` to the sum
   - This creates a unique hash for each pattern type

2. **Index Mapping**: `%9`
   - Takes modulo 9 of the hash
   - Maps the four pattern hashes to distinct indices: 0, 2, 6, 7

3. **Value Extraction**: `393>> ... &7`
   - The magic number 393 = 0b110001001 encodes all four output values
   - Bit positions: 0->1, 2->2, 6->6, 7->3
   - Right-shifts by the index, then masks with 7 to extract 3 bits

## Key Insights

1. **Hash Function Design**: The combination of multiplier (7) and modulo (9) was found through exhaustive search to give the smallest possible bit-packed representation.

2. **Bit Packing**: Instead of a lookup table, encoding values in a single integer saves bytes when the values are small enough to fit in consecutive bit fields.

3. **3-bit Mask**: Since the maximum output value is 6 (binary 110), we can use `&7` instead of `&15`, saving one byte.
