def solve(g):
 R,C=len(g),len(g[0]);o=[[0]*C for _ in range(R)];M=[]
 for i in range(R):
  for j in range(C):
   if g[i][j]:M+=[(i,j,g[i][j])]
 M.sort(key=lambda x:x[0]*C+x[1])
 r1,c1,v1=M[0];r2,c2,v2=M[1]
 rd,cd=r2-r1,c2-c1
 if c1==c2 or(rd>0 and rd<cd):
  d=rd
  for i in range(R):
   for j in range(C):
    k=(i-r1)%(2*d)
    if i>=r1:
     if k==0:o[i][j]=v1
     elif k==d:o[i][j]=v2
 else:
  d=cd
  for i in range(R):
   for j in range(C):
    k=(j-c1)%(2*d)
    if j>=c1:
     if k==0:o[i][j]=v1
     elif k==d:o[i][j]=v2
 return o
