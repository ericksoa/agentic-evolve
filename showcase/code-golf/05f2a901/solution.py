def solve(g):
 P=[(i,j,v)for i,q in enumerate(g)for j,v in enumerate(q)if v];F=lambda n,k:sorted(p[k]for p in P if p[2]==n);R,r,C,c=F(2,0),F(8,0),F(2,1),F(8,1);H=lambda a,b:(b[0]>a[-1])*(b[0]-a[-1]-1)+(a[0]>b[-1])*(b[-1]-a[0]+1);d=H(R,r);e=H(C,c);o=[[0]*len(g[0])for w in g]
 for i,j,v in P:o[i+(v<8)*d][j+(v<8)*e]=v
 return o