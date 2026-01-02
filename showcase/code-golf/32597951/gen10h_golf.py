def solve(g):
 o=[*map(list,g)];e={(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v>7}
 [o[r].__setitem__(c,3)for r,w in enumerate(g)for c,v in enumerate(w)if v==1and min(i for i,_ in e)<=r<=max(i for i,_ in e)and min(j for _,j in e)<=c<=max(j for _,j in e)and e&{(r+a,c+b)for a in(-1,0,1)for b in(-1,0,1)if a|b}]if e else 0
 return o
