def solve(g,I=range):
 o=eval(str(g))
 for r in I(len(g)-2):
  c=0
  while c<len(w:=g[r]):
   v=w[c];e=c
   while w[e:e+1]==[v]:e+=1
   if v and w[c:e]==g[r+1][c:e]==g[r+2][c:e]:
    for x in I(c+1,e,2):o[r+1][x]=0
   c=e or c+1
 return o
