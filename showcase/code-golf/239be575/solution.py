def solve(g):
 (r,c),(R,C)=[(i,j)for i,(a,b)in enumerate(zip(g,g[1:]))for j in range(len(a)-1)if a[j]&a[j+1]&b[j]&b[j+1]>1]
 return[[8*(C>c<16>g[r+1][C+1]+g[R+1][c+1])]]
