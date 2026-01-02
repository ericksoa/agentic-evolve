def solve(g):
 R,C=len(g),len(g[0]);o=[r[:]for r in g]
 for i in range(R):
  for j in range(C):
   if g[i][j]==2:
    for k in range(R):
     if o[k][j]==0:o[k][j]=2
 for i in range(R):
  for j in range(C):
   v=g[i][j]
   if v in(1,3):
    for k in range(C):
     if o[i][k] in(0,2):o[i][k]=v
 return o
