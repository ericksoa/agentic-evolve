def solve(g):
 R,C=len(g),len(g[0]);c=[]
 for r in range(R):
  for j in range(C):
   if g[r][j]:c+=[(r,j,g[r][j])]
 c.sort();r1,_,c1=c[0];r2,_,c2=c[1];o=[[0]*C for _ in range(R)]
 for i in range(R):
  for j in range(C):
   if i<(r1+r2)//2+1:
    if i==0 or i==r1 or i==R-1:o[i][j]=c1
    elif j==0 or j==C-1:o[i][j]=c1
   else:
    if i==r2 or i==R-1:o[i][j]=c2
    elif j==0 or j==C-1:o[i][j]=c2
 return o