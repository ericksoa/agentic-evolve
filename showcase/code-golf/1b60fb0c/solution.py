def solve(g):
 R,C=len(g),len(g[0]);o=[r[:]for r in g]
 mc=min(j for i in range(R)for j in range(C)if g[i][j]==1)
 # Find mxc (max extension) from connected rows
 mxc=0
 for i in range(R):
  ones=[j for j in range(C)if g[i][j]==1]
  if ones:
   minc,maxc=min(ones),max(ones)
   if len(ones)==maxc-minc+1 or maxc-minc-len(ones)<=1:
    mxc=max(mxc,maxc)
 # Find extended rows
 exr=set()
 for i in range(R):
  ones=[j for j in range(C)if g[i][j]==1]
  if ones:
   minc,maxc=min(ones),max(ones)
   if (len(ones)==maxc-minc+1 or maxc-minc-len(ones)<=1)and maxc>=mxc:
    exr.add(i)
 if exr:
  mr,Mr=min(exr),max(exr)
  for i in range(mr,Mr+2):
   if i<R and any(g[i][j]==1 for j in range(mc,mc+3)):
    exr.add(i)
 # Mirror around leftmost 1 in each row
 for i in exr:
  ones=[j for j in range(C)if g[i][j]==1]
  if ones:
   ax=min(ones)
   for j in ones:
    if j>ax:
     mj=2*ax-j
     if 0<mj<ax and o[i][mj]==0:o[i][mj]=2
 return o
