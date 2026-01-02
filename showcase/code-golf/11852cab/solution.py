def solve(g):
 R,C=len(g),len(g[0]);o=[r[:]for r in g];P=[(i,j)for i in range(R)for j in range(C)if g[i][j]]
 if P:
  I,J=zip(*P);z=(min(I)+max(I))/2+1j*(min(J)+max(J))/2
  for i,j in P:
   d=i+1j*j-z
   for r in 1,1j,-1,-1j:
    p=z+d*r;pi,pj=int(p.real),int(p.imag)
    if R>pi>=0<=pj<C and o[pi][pj]<1:o[pi][pj]=g[i][j]
 return o
