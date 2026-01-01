# Evolution Log: Task a64e4611

## Task Analysis
- **Pattern**: Find the largest rectangular region of zeros in a 30x30 grid
- **Rule**:
  1. Find largest all-zero rectangle using maximal rectangle in histogram algorithm
  2. Shrink based on orientation (horizontal vs vertical)
  3. Fill core with 3s
  4. Extend rows/columns to edges if ALL neighbors also extend to edges
- **Difficulty**: Hard (complex algorithm, neighbor-conditional extensions)

## Algorithm Discovery

This task required significant reverse-engineering to understand. Initial hypotheses failed:
- Simple maximal rectangle - Wrong (expected output is a cross, not rectangle)
- Flood fill - Wrong (connected component doesn't match)
- Direct rectangle fill - Wrong (too many cells filled)

**Key insight**: The algorithm produces a **cross pattern** by:
1. Finding largest zero rectangle
2. Shrinking the core based on orientation
3. Extending arms only where both neighbors extend to edges

## Evolution Results

### Baseline (Gen 0)
```python
# Initial working solution - ~1200 bytes
# (Verbose with separate functions for each step)
```
- **Bytes**: ~1200
- **Score**: ~1300

### Champion (Gen 65 - Final)
```python
def solve(G):
 R,C=len(G),len(G[0]);O=[*map(list,G)];h=[];b=0,
 for r in range(R):
  h+=[(1+h[r-1][c]if r else 1)*(O[r][c]<1)for c in range(C)],;s=[]
  for c in range(C+1):
   t=h[r][c]if c<C else 0;x=c
   while s and s[-1][1]>t:q,w=s.pop();b=max(b,(w*(c-q),r-w+1,q,r,c-1));x=q
   s+=(x,t),
 if b[0]<1:return G
 _,e,f,g,j=b;H=j-f>g-e;e+=e>0;g-=H*(g<R-1);f+=1-H;j-=1-H
 for r in range(e,g+1):G[r][f:j+1]=[3]*(j-f+1)
 E=lambda i,L,z:all(O[(v,i)[z]][(i,v)[z]]<1for v in L)
 for i,A,B,P,z in[(r,e,g,[(range(f),f),(range(j+1,C),j<C-1)],1)for r in range(e,g+1)]+[(c,f,j,[(range(e),e),(range(g+1,R),g<R-1)],0)for c in range(f,j+1)]:
  for L,k in P:
   if k*E(i,L,z)*(i==A or E(i-1,L,z))*(i==B or E(i+1,L,z)):
    for v in L:G[(v,i)[z]][(i,v)[z]]=3
 return G
```
- **Bytes**: 750
- **Score**: 1750
- **Improvement**: ~450 bytes (37.5% reduction)

## Key Innovations

1. **Merged loops**: Histogram building + max rect finding in single row loop
2. **Trailing comma append**: `h+=[...],` instead of `h+=[[...]]` saves 1 byte
3. **`max()` for best rect**: `b=max(b,(area,coords))` instead of conditional
4. **Multiplication for boolean logic**: `*(O[r][c]<1)` instead of `and`
5. **Unified shrink formula**: `e+=e>0;g-=H*(g<R-1);f+=1-H;j-=1-H` based on H (horizontal flag)
6. **Unified E lambda**: `O[(v,i)[z]][(i,v)[z]]` uses tuple indexing for row/col swapping
7. **Slice assignment**: `G[r][f:j+1]=[3]*(j-f+1)` for core fill
8. **`[*map(list,G)]`**: Shorter than `[r[:]for r in G]` for deep copy
9. **Combined extension loop**: Single comprehension for both row and column extensions

## Evolution Progression

| Gen | Bytes | Key Change |
|-----|-------|------------|
| 0 | ~1200 | Initial working solution |
| 10 | ~1000 | Inlined helper functions |
| 20 | 858 | Unified E lambda, max() for comparison |
| 30 | 825 | Removed N function, inlined neighbor check |
| 40 | 794 | Tuple indexing for E lambda |
| 50 | 763 | Merged histogram + max rect loops |
| 55 | 759 | Trailing comma append, b[0]<1 check |
| 60 | 751 | Unified shrink with H multiplication |
| 65 | 750 | `[*map(list,G)]` for copy |

## Techniques Attempted

### Successful
- Merged loops for histogram + max rect
- `max()` instead of conditional comparison
- Trailing comma for list append
- Tuple indexing for dimension swapping
- Slice assignment for core fill
- Unified shrink formula with multiplication
- `[*map(list,G)]` for deep copy

### Failed to Improve
- Pre-computed range variables (added overhead)
- `__setitem__` for fills (longer)
- `exec()` with string (more complex)
- Separate helper functions (more bytes)
- Bitwise operations (requires parentheses)
- Alternative algorithms (all longer)

## Compressed Version

A zlib-compressed version achieves 664 bytes:
```python
import zlib as z,base64 as b
exec(z.decompress(b.b64decode("...")))
```
(Not counted as "real" golfing but demonstrates compression potential)

## Summary

- **Starting**: ~1200 bytes (verbose with helpers)
- **Final**: 750 bytes (highly optimized single function)
- **Improvement**: ~450 bytes (37.5% reduction)
- **Mutations tested**: 65+
- **Plateau reached**: Multiple optimizations at 750 bytes, no shorter discovered
- **Difficulty**: Significantly harder than typical ARC tasks due to complex algorithm
