def solve(g):
 I,J=zip(*[(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v])
 return[r[min(J):max(J)+1]for r in g[min(I):max(I)+1]]
