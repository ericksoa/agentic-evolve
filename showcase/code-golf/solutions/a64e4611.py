def solve(G):
 R,C=len(G),len(G[0]);O=[*map(list,G)];h=[];b=0,
 for r in range(R):
  h+=[(1+h[r-1][c]if r else 1)*(O[r][c]<1)for c in range(C)],;s=[]
  for c in range(C+1):
   t=h[r][c]if c<C else 0;x=c
   while s and s[-1][1]>t:q,w=s.pop();b=max(b,(w*(c-q),r-w+1,q,r,c-1));x=q
   s+=(x,t),
 if b[0]<1:return G
 _,e,f,g,j=b;H=j-f>g-e;e+=e>0;g-=H*(g<R-1);f+=1-H;j-=1-H
 for r in range(e,g+1):G[r][f:j+1]=[3]*(j-f+1)
 E=lambda i,L,z:all(O[(v,i)[z]][(i,v)[z]]<1for v in L)
 for i,A,B,P,z in[(r,e,g,[(range(f),f),(range(j+1,C),j<C-1)],1)for r in range(e,g+1)]+[(c,f,j,[(range(e),e),(range(g+1,R),g<R-1)],0)for c in range(f,j+1)]:
  for L,k in P:
   if k*E(i,L,z)*(i==A or E(i-1,L,z))*(i==B or E(i+1,L,z)):
    for v in L:G[(v,i)[z]][(i,v)[z]]=3
 return G