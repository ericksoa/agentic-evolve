def solve(g):
 R=len(g);C=len(g[0])
 return max(((H-r)*(W-c),[x[c+1:W-1]for x in g[r+1:H-1]])for r in range(R)for c in range(C)for H in range(r+2,R+1)for W in range(c+2,C+1)if 2>len({*g[r][c:W]+g[H-1][c:W],*[z[k]for z in g[r:H]for k in(c,W-1)]}))[1]