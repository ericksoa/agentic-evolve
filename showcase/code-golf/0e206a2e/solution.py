def solve(g):
 h,w=len(g),len(g[0]);p={}
 for r in range(h):
  for c in range(w):
   if v:=g[r][c]:p[v]=p.get(v,[])+[(r,c)]
 f=max(p,key=lambda k:len(p[k]));V=set();C=[]
 def B(s):
  q=[s];c=set()
  while q:
   r,z=q.pop()
   if(r,z)in V:continue
   V.add((r,z))
   if g[r][z]:
    c.add((r,z))
    for a in-1,0,1:
     for b in-1,0,1:
      if a|b and h>r+a>=0<=z+b<w and(r+a,z+b)not in V and g[r+a][z+b]:q+=(r+a,z+b),
  return c
 for x in p[f]:
  if x not in V:c=B(x);C+=[c]*bool(c)
 I={k:{*p[k]}-set().union(*C)for k in p if k!=f};o=[[0]*w for _ in g];u=set()
 for c in C:
  K={g[r][z]:(r,z)for r,z in c if g[r][z]!=f}
  if K:
   R,*_=K;Rr,Rc=K[R];L=[(r-Rr,z-Rc,g[r][z])for r,z in c];X=[(j,r-Rr,z-Rc)for j,(r,z)in K.items()if j!=R]
   for a,b,d,t in(1,0,0,1),(-1,0,0,1),(1,0,0,-1),(-1,0,0,-1),(0,1,1,0),(0,-1,1,0),(0,1,-1,0),(0,-1,-1,0):
    if R in I:
     for ir,ic in I[R]:
      if(ir,ic)not in u and all(cv in I and(ir+r*a+z*b,ic+r*d+z*t)in I[cv]for cv,r,z in X):
       for r,z,j in L:
        if h>ir+r*a+z*b>=0<=ic+r*d+z*t<w:o[ir+r*a+z*b][ic+r*d+z*t]=j
       u|={(ir,ic)}|{(ir+r*a+z*b,ic+r*d+z*t)for cv,r,z in X}
 return o
