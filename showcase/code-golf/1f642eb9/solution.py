def solve(g):
 o=eval(str(g));a=b=c=d=5;M=[]
 for r,R in enumerate(g):
  for j,v in enumerate(R):
   if v==8:a=min(a,r);b=max(b,r);c=min(c,j);d=max(d,j)
   elif v:M+=(r,j,v),
 for r,j,v in M:
  if c<=j<=d:o[(a,b)[r>b]][j]=v
  if a<=r<=b:o[r][(c,d)[j>d]]=v
 return o
