def solve(g):
 R,C=len(g),len(g[0]);o=[r[:]for r in g];e=[(r,c)for r in range(R)for c in range(C)if g[r][c]==8]
 if e:
  for r in range(min(r for r,c in e),max(r for r,c in e)+1):
   for c in range(min(c for r,c in e),max(c for r,c in e)+1):
    if g[r][c]==1and any(0<=r+d<R>0<=c+f<C and g[r+d][c+f]==8for d in(-1,0,1)for f in(-1,0,1)if d|f):o[r][c]=3
 return o
