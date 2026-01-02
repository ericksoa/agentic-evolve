def solve(g):
 R,C=len(g),len(g[0]);m,s,k=set(),set(),0
 for r,w in enumerate(g):
  for c,v in enumerate(w):
   if v==2:m|={(r,c)}
   elif v:s|={(r,c)};k=v
 mr={r for r,c in m};mc={c for r,c in m};sr={r for r,c in s};sc={c for r,c in s}
 o=[[3]*C for _ in g]
 for r,c in s|m:o[r][c]=k
 if len(mr)<2:
  M,=mr;e=M+.5if min(sr)>M else M-.5
  for r,c in s:n=int(2*e-r);R>n>=0and(o[n].__setitem__(c,k))
 elif len(mc)<2:
  M,=mc;e=M-.5if max(sc)<M else M+.5
  for r,c in s:n=int(2*e-c);C>n>=0and(o[r].__setitem__(n,k))
 return o
