def solve(g):
 R,C=len(g),len(g[0]);o=[[0]*C for _ in range(R)];V={};H={}
 for c in range(C):
  for v in range(1,10):
   if sum(g[r][c]==v for r in range(R))>R//2:V[v]=c
 for r in range(R):
  for v in range(1,10):
   if sum(g[r][c]==v for c in range(C))>C//2:H[v]=r
 for i in range(R):
  for j in range(C):
   v=g[i][j]
   if v in V and j==V[v]:o[i][j]=v
   if v in H and i==H[v]:o[i][j]=v
 for i in range(R):
  for j in range(C):
   v=g[i][j]
   if v<1:continue
   if v in V and j!=V[v]:
    n=V[v]+(1if j>V[v]else-1)
    if 0<=n<C:o[i][n]=v
   elif v in H and i!=H[v]:
    n=H[v]+(1if i>H[v]else-1)
    if 0<=n<R:o[n][j]=v
 return o
