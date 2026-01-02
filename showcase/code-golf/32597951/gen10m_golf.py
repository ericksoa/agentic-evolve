def solve(g):
 o=[*map(list,g)]
 (e:={(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v>7})and(I:=list(zip(*e)))and[o[r].__setitem__(c,3)for r in range(min(I[0]),max(I[0])+1)for c in range(min(I[1]),max(I[1])+1)if g[r][c]==1if e&{(r+a,c+b)for a in(-1,0,1)for b in(-1,0,1)if a|b}]
 return o
