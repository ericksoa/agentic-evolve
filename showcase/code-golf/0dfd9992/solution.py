def solve(g,E=enumerate):
 for y in range(1,-~len(g)):
  for x in range(1,-~len(g[0])):
   m={}
   if all(v<1or v==m.setdefault((i%y,j%x),v)for i,r in E(g)for j,v in E(r))>y*x-len(m):return[[v or m[i%y,j%x]for j,v in E(r)]for i,r in E(g)]
