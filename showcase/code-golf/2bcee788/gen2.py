# Gen 2: Remove whitespace, use list comprehensions
def solve(g):
 R,C=len(g),len(g[0]);m,s,k=set(),set(),0
 for r in range(R):
  for c in range(C):
   if g[r][c]==2:m.add((r,c))
   elif g[r][c]:s.add((r,c));k=g[r][c]
 mr={r for r,c in m};mc={c for r,c in m};sr={r for r,c in s};sc={c for r,c in s}
 o=[[3]*C for _ in range(R)]
 for r,c in s|m:o[r][c]=k
 if len(mr)==1:
  M,=mr;e=M+.5if min(sr)>M else M-.5
  for r,c in s:
   n=int(2*e-r)
   if 0<=n<R:o[n][c]=k
 elif len(mc)==1:
  M,=mc;e=M-.5if max(sc)<M else M+.5
  for r,c in s:
   n=int(2*e-c)
   if 0<=n<C:o[r][n]=k
 return o
