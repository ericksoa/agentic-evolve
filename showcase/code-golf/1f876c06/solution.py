def solve(g):
 for v in{*sum(g,[])}-{0}:(a,b),(c,d)=[(i,j)for i,r in enumerate(g)for j,c in enumerate(r)if c==v];exec("g[a][b]=v;a+=(c>a)-(a>c);b+=(d>b)-(b>d);"*9)
 return g
