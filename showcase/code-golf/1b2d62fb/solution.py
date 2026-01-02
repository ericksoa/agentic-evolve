def solve(g):
 R=len(g);c=3;W=len(g[0])-c-1
 o=[[0]*W for _ in range(R)]
 for i in range(R):
  for j in range(W):
   if g[i][j]==0 and g[i][c+1+j]==0:o[i][j]=8
 return o
