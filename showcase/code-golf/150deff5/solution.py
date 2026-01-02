def solve(g):
 R,C=len(g),len(g[0]);F={divmod(x,C)for x in range(R*C)if g[x//C][x%C]>4};V=[(i,j)for i in range(R-1)for j in range(C-1)if{(i,j),(i+1,j),(i,j+1),(i+1,j+1)}<=F];n=len(V)
 def f(m):s=[V[k]for k in range(n)if m>>k&1];c={(i+a,j+b)for i,j in s for a in(0,1)for b in(0,1)};T=F-c;return(len(c)!=4*len(s))*9or sum(not T&{(i-1,j),(i+1,j),(i,j-1),(i,j+1)}for i,j in T),~len(s),c
 E=min(f(m)for m in range(1<<n))[2]
 return[[8*((i,j)in E)or 2*((i,j)in F)for j in range(C)]for i in range(R)]
