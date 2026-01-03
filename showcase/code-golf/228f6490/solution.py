def solve(g):
 o=eval(str(g));R=len(g);C=len(g[0])
 for v in{*sum(g,[])}-{0,5}:
  P=[(r,c)for r,w in enumerate(g)for c,x in enumerate(w)if x==v];H=sorted((r-P[0][0],c-P[0][1])for r,c in P)
  [exec("for e,f in H:o[a+1+e][b+1+f]=v\nfor r,k in P:o[r][k]=0")for a in range(R)for b in range(C)for c in range(a+2,R)for d in range(b+2,C)if{5}=={*g[a][b:d+1],*g[c][b:d+1]}|{g[y][b]for y in range(a,c+1)}|{g[y][d]for y in range(a,c+1)}and sorted((r-a-1,k-b-1)for r in range(a+1,c)for k in range(b+1,d)if g[r][k]<1)==H]
 return o
