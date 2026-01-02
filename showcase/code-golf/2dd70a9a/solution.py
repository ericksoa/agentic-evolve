def solve(g):
 R,C=len(g),len(g[0]);o=[r[:]for r in g];p2=[(i,j)for i in range(R)for j in range(C)if g[i][j]==2];p3=[(i,j)for i in range(R)for j in range(C)if g[i][j]==3]
 if not p2 or not p3:return o
 r2,c2=sorted(set(i for i,j in p2)),sorted(set(j for i,j in p2));r3,c3=sorted(set(i for i,j in p3)),sorted(set(j for i,j in p3));G=c2[0]-c3[-1];D=abs(r2[0]-r3[0])
 if G<3:
  V=max(c2[-1],c3[-1])+(G if G>0 else D+1)
  for j in range(c2[-1]+1,V+1):
   if j<C and o[r2[0]][j]<1:o[r2[0]][j]=3
  for j in range(c3[-1]+1,V+1):
   if j<C and o[r3[0]][j]<1:o[r3[0]][j]=3
  for i in range(min(r2[0],r3[0])+1,max(r2[-1],r3[-1])):
   if V<C and o[i][V]<1:o[i][V]=3
 else:
  K=(r2[-1]+(r3[0]-r2[-1])//3)if r3[0]>r2[-1]else(r3[-1]+(r2[0]-r3[-1])//3);a,b=(c2[-1],c3[0])if r3[0]>r2[-1]else(c3[0],c2[-1])
  for i in range((r2[-1]if r3[0]>r2[-1]else r3[-1])+1,K+1):
   if o[i][a if r3[0]>r2[-1]else c3[0]]<1:o[i][a if r3[0]>r2[-1]else c3[0]]=3
  for j in range(min(c2[-1],c3[0]),max(c2[-1],c3[0])+1):
   if o[K][j]<1:o[K][j]=3
  for i in range(K+1,(r3[0]if r3[0]>r2[-1]else r2[0])):
   if o[i][c3[0]if r3[0]>r2[-1]else c2[-1]]<1:o[i][c3[0]if r3[0]>r2[-1]else c2[-1]]=3
 return o
