def solve(g):
 p=[r[:3]for r in g[:3]];o=[r[:]for r in g]
 for i,r in enumerate(g):
  for j,v in enumerate(r):
   if v==1:
    for y in range(3):
     for x in range(3):o[i//3*3+y][j-1+x]=p[y][x]
 return o