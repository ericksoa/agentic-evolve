def solve(g):
 R,C=len(g),len(g[0])
 o=[r[:]for r in g]
 e=[(r,c)for r in range(R)for c in range(C)if g[r][c]==8]
 if not e:return o
 r0,r1=min(r for r,c in e),max(r for r,c in e)
 c0,c1=min(c for r,c in e),max(c for r,c in e)
 for r in range(r0,r1+1):
  for c in range(c0,c1+1):
   if g[r][c]==1 and any(g[r+d][c+e]==8for d in(-1,0,1)for e in(-1,0,1)if 0<=r+d<R and 0<=c+e<C and(d or e)):o[r][c]=3
 return o
