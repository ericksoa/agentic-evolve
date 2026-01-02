def solve(g):
 o=[*map(list,g)];e={(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v>7};R,C=zip(*e)if e else([],[])
 for r in range(min(R,default=0),max(R,default=0)+1):
  for c in range(min(C),max(C)+1):
   if g[r][c]==1and{(r+a,c+b)for a in(-1,0,1)for b in(-1,0,1)if a|b}&e:o[r][c]=3
 return o
