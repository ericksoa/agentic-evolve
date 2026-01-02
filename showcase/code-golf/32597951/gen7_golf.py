def solve(g):
 o=[r[:]for r in g];e={(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v==8};I,J=zip(*e)if e else((),())
 for r in range(min(I),max(I)+1)if e else[]:
  for c in range(min(J),max(J)+1):
   if g[r][c]==1and e&{(r+a,c+b)for a in(-1,0,1)for b in(-1,0,1)if a|b}:o[r][c]=3
 return o
