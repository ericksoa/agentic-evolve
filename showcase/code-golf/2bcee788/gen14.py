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
  M,=mr;e=2*M+1-2*(min(r for r,c in s)<=M)
  for r,c in s:n=e-r;R>n>=0and o[n].__setitem__(c,k)
 elif L(mc)<2:
  M,=mc;e=2*M-1+2*(max(c for r,c in s)>=M)
  for r,c in s:n=e-c;C>n>=0and o[r].__setitem__(n,k)
 return o
