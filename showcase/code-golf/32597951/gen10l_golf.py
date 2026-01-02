def solve(g):
 o=[*map(list,g)];e={(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v>7}
 if e:I,J=zip(*e);[o[r].__setitem__(c,3)for r in range(min(I),max(I)+1)for c in range(min(J),max(J)+1)if g[r][c]-1==0and e&{(r+a,c+b)for a in(-1,0,1)for b in(-1,0,1)if a|b}]
 return o
