def solve(g):
 o=[*map(list,g)];e={(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v>7}
 e and(I:=zip(*e),J:=zip(*e),[o[r].__setitem__(c,3)for r in range(min(i for i,j in e),max(i for i,j in e)+1)for c in range(min(j for i,j in e),max(j for i,j in e)+1)if g[r][c]==1if e&{(r+a,c+b)for a in(-1,0,1)for b in(-1,0,1)if a|b}])
 return o
