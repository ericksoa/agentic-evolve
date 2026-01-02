# ARC Task 2bcee788 - Color Replacement by Marker

## Pattern: Reflection Across Marker Line

The transformation:
1. Background (0) becomes green (3)
2. Markers (2) become the shape color
3. Shape is reflected across the marker line (the edge between marker and shape cells)

## Evolution Log

| Gen | Bytes | Score | Key Changes |
|-----|-------|-------|-------------|
| 0   | 1440  | 1060  | Initial working solution - verbose, readable |
| 1   | 778   | 1722  | Shortened variable names (R,C,m,s,k,mr,mc,sr,sc,o) |
| 2   | 582   | 1918  | Remove whitespace, use set union `s\|m` |
| 3   | 564   | 1936  | Use `m\|={(r,c)}` syntax, compact conditionals |
| 4   | 534   | 1966  | Remove comments, streamline unpacking |
| 5   | 527   | 1973  | Use `for _ in g` instead of `range(R)` |
| 6   | 523   | 1977  | Use `R>n>=0` chain, `len(mr)<2` |
| 7   | 517   | 1983  | Use lists with `+=` instead of sets, tuple append `+=(r,c),` |
| 8   | 511   | 1989  | Inline min/max in edge calculation |
| 9   | 509   | 1991  | Use `m+=(r,c),` tuple append syntax |
| 10  | 501   | 1999  | Use `for p in m+s:o[p[0]][p[1]]=k` |
| 11  | 497   | 2003  | Use `for a,b in m+s:o[a][b]=k` |
| 12  | 495   | 2005  | Alias `L=len` and reuse |
| 13  | 487   | 2013  | Simplify edge formula using `2*M+.5*(2*(...)-1)` |
| 14  | 469   | 2031  | Convert to integer math: `e=2*M+1-2*(condition)` |
| 15  | 465   | 2035  | Inline `n` variable: `R>e-r>=0and o[e-r].__setitem__(c,k)` |

## Champion Solution (465 bytes)

```python
def solve(g):
 L=len;R,C=L(g),L(g[0]);m,s,k=[],[],0
 for r,w in enumerate(g):
  for c,v in enumerate(w):
   if v==2:m+=(r,c),
   elif v:s+=(r,c),;k=v
 mr={r for r,c in m};mc={c for r,c in m}
 o=[[3]*C for _ in g]
 for a,b in m+s:o[a][b]=k
 if L(mr)<2:
  M,=mr;E=2*M+1-2*(min(r for r,c in s)<=M)
  for r,c in s:R>E-r>=0and o[E-r].__setitem__(c,k)
 elif L(mc)<2:
  M,=mc;E=2*M-1+2*(max(c for r,c in s)>=M)
  for r,c in s:C>E-c>=0and o[r].__setitem__(E-c,k)
 return o
```

## Key Golf Tricks Used

1. **Tuple append syntax**: `m+=(r,c),` instead of `m.append((r,c))`
2. **Chained unpacking**: `M,=mr` to extract single element from set
3. **Boolean to integer**: `2*(condition)-1` gives -1 or 1
4. **Chained comparison**: `R>E-r>=0` instead of `0<=E-r<R`
5. **__setitem__ trick**: `o[r].__setitem__(c,k)` to avoid assignment in expression
6. **Len alias**: `L=len` saves bytes when used multiple times
7. **List comprehension as iterator**: `for _ in g` instead of `for _ in range(len(g))`

## Score Calculation

- Byte count: 465
- Score formula: max(1, 2500 - byte_count) = 2500 - 465 = **2035 points**
