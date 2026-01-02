def solve(g):
 R,C=len(g),len(g[0]);o=[[0]*C for _ in range(R)];S={}
 for i in range(R):
  for j in range(C):
   v=g[i][j]
   if v:S.setdefault(v,[]).append((i,j))
 tr=min(i for i,j in S[1])
 for v,P in S.items():
  sr=min(i for i,j in P)
  for i,j in P:o[i-sr+tr][j]=v
 return o
