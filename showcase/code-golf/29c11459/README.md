# ARC Task 29c11459 - Horizontal Line Splitting

## Pattern Description
Find rows with non-zero values at both endpoints (first and last columns). For each such row:
- Fill the left half with the left endpoint value
- Place 5 in the center
- Fill the right half with the right endpoint value

Grid width is always 11, so mid = 5.

## Champion Solution
```python
solve=lambda g:[w[0]*w[-1]and[w[0]]*5+[5]+[w[-1]]*5or w for w in g]
```

**Bytes: 68** | **Score: 2432**

## Evolution Log

### Generation 0 (365 bytes)
Initial readable solution with full variable names and proper indentation.
```python
def solve(grid):
    result = []
    for row in grid:
        if row[0] != 0 and row[-1] != 0:
            left = row[0]
            right = row[-1]
            width = len(row)
            mid = width // 2
            new_row = [left] * mid + [5] + [right] * mid
            result.append(new_row)
        else:
            result.append(row[:])
    return result
```

### Generation 1 (119 bytes)
Applied basic golf: single-char variable names, minimal whitespace, removed unnecessary parentheses.
```python
def solve(g):
 r=[]
 for w in g:
  if w[0]and w[-1]:m=len(w)//2;r+=[[w[0]]*m+[5]+[w[-1]]*m]
  else:r+=[w[:]]
 return r
```

### Generation 2 (93 bytes)
Converted to list comprehension with ternary operator.
```python
def solve(g):return[[w[0]]*(m:=len(w)//2)+[5]+[w[-1]]*m if w[0]and w[-1]else w[:]for w in g]
```

### Generation 3 (90 bytes)
Replaced `w[0]and w[-1]` with `w[0]*w[-1]` (both are truthy when both non-zero).
```python
def solve(g):return[[w[0]]*(m:=len(w)//2)+[5]+[w[-1]]*m if w[0]*w[-1]else w[:]for w in g]
```

### Generation 4 (86 bytes)
Converted `def` to `lambda` for shorter function definition.
```python
solve=lambda g:[[w[0]]*(m:=len(w)//2)+[5]+[w[-1]]*m if w[0]*w[-1]else w[:]for w in g]
```

### Generation 5 (86 bytes)
Tried `w+[]` instead of `w[:]` - same length (no improvement).

### Generation 6 (84 bytes)
Removed copy (`w[:]` to `w`) since we don't modify input rows.
```python
solve=lambda g:[[w[0]]*(m:=len(w)//2)+[5]+[w[-1]]*m if w[0]*w[-1]else w for w in g]
```

### Generation 7 (82 bytes)
Used `and/or` pattern instead of ternary `if/else`.
```python
solve=lambda g:[w[0]*w[-1]and[w[0]]*(m:=len(w)//2)+[5]+[w[-1]]*m or w for w in g]
```

### Generation 8 (68 bytes) - CHAMPION
Hardcoded width=5 since all grids are 11 columns wide (mid=5). This is a significant optimization exploiting task-specific constraints.
```python
solve=lambda g:[w[0]*w[-1]and[w[0]]*5+[5]+[w[-1]]*5or w for w in g]
```

### Generations 9-10 (68 bytes)
Tried various mutations that all proved longer:
- Walrus operators for a/b assignment (+3 bytes)
- `5*[x]` prefix instead of `[x]*5` (+1 byte with space)
- List unpacking `[*5*[w[0]],5,*5*[w[-1]]]` (+2 bytes)
- map/lambda approach (+6 bytes)

## Key Golf Techniques Used
1. **Lambda over def** - saves ~4 bytes
2. **and/or over ternary** - saves 2 bytes
3. **Multiplication for condition** - `w[0]*w[-1]` instead of `w[0]and w[-1]`
4. **Task-specific optimization** - hardcoded width exploits consistent grid size
5. **Skip unnecessary copy** - `w` instead of `w[:]` when not modifying
