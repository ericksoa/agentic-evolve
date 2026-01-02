def solve(g):
 o=[r[:]for r in g];e=[(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v==8]
 if e:
  for r in range(min(i for i,j in e),max(i for i,j in e)+1):
   for c in range(min(j for i,j in e),max(j for i,j in e)+1):
    if g[r][c]==1and any((r+a,c+b)in e for a in(-1,0,1)for b in(-1,0,1)if a|b):o[r][c]=3
 return o
