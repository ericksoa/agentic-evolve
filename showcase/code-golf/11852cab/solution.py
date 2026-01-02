def solve(g):
 R,C=len(g),len(g[0]);o=eval(str(g));P=[(i,j)for i in range(R)for j in range(C)if g[i][j]]
 if P:I,J=zip(*P);s,t=min(I)+max(I),min(J)+max(J);u=s+t>>1
 for i,j in P:
  for a,b in(u-j,t+i-u),(s-i,t-j),(j+u-t,u-i):
   if R>a>=0<=b<C:o[a][b]=o[a][b]or g[i][j]
 return o
