def solve(g):
 R,C=len(g),len(g[0]);o=[[0]*C for _ in range(R)]
 for i in range(R):
  for j in range(C):
   if g[i][j]==5:
    r,c=i%2,j%2
    if r==0 and c==0:
     if i+1<R and j+1<C and g[i][j+1]==5 and g[i+1][j]==5 and g[i+1][j+1]==5:o[i][j]=o[i][j+1]=o[i+1][j]=o[i+1][j+1]=8
     else:o[i][j]=2
    elif r==0 and c==1:
     if i+1<R and j-1>=0 and g[i][j-1]==5 and g[i+1][j]==5 and g[i+1][j-1]==5:pass
     else:o[i][j]=2
    elif r==1 and c==0:
     if i-1>=0 and j+1<C and g[i-1][j]==5 and g[i][j+1]==5 and g[i-1][j+1]==5:pass
     else:o[i][j]=2
    else:
     if i-1>=0 and j-1>=0 and g[i-1][j]==5 and g[i][j-1]==5 and g[i-1][j-1]==5:pass
     else:o[i][j]=2
 return o
