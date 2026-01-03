def solve(g):
 H,W=len(g),len(g[0]);S=set();A=[];R=range;D=-1,0,1;C={}
 for r in R(H):
  for c in R(W):
   if g[r][c]and(r,c)not in S:
    v,s,P=g[r][c],[(r,c)],[]
    while s:
     i,j=s.pop()
     if H>i>=0<=j<W and(i,j)not in S and g[i][j]==v:P+=(i,j),;S|={(i,j)};s+=[(i+a,j+b)for a in D for b in D if a|b]
    Y,X=zip(*P);n=frozenset((r-min(Y),c-min(X))for r,c in P);C[n]=C.get(n,0)+1;A+=(n,v,P),
 _,v,P=max(A,key=lambda x:C[x[0]]);Y,X=zip(*P)
 return[[v*((r,c)in P)for c in R(min(X),max(X)+1)]for r in R(min(Y),max(Y)+1)]