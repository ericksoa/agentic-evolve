def solve(g):
 R,C=len(g),len(g[0]);o=[r[:]for r in g];e={c+r*C for r in range(R)for c in range(C)if g[r][c]==8}
 if e:
  for p in range(R*C):
   r,c=p//C,p%C
   if min(e)//C<=r<=max(e)//C and min(x%C for x in e)<=c<=max(x%C for x in e)and g[r][c]==1and any(p+d in e for d in(-C-1,-C,-C+1,-1,1,C-1,C,C+1)):o[r][c]=3
 return o
