def solve(g):
 L=len;R,C=L(g),L(g[0]);m,s,k=[],[],0
 for r,w in enumerate(g):
  for c,v in enumerate(w):
   if v==2:m+=(r,c),
   elif v:s+=(r,c),;k=v
 mr={r for r,c in m};mc={c for r,c in m}
 o=[[3]*C for _ in g]
 for a,b in m+s:o[a][b]=k
 if L(mr)<2:
  M,=mr;e=M+.5if min(r for r,c in s)>M else M-.5
  for r,c in s:n=int(2*e-r);R>n>=0and o[n].__setitem__(c,k)
 elif L(mc)<2:
  M,=mc;e=M-.5if max(c for r,c in s)<M else M+.5
  for r,c in s:n=int(2*e-c);C>n>=0and o[r].__setitem__(n,k)
 return o
