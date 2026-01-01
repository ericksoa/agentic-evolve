def solve(G):
 I=range;R,C=len(G),len(G[0]);O=[*map(list,G)];h=[];b=0,
 for r in I(R):
  h+=[(r and 1+h[r-1][c]or 1)*(O[r][c]<1)for c in I(C)],;s=[]
  for c in I(C+1):
   t=c<C and h[r][c];x=c
   while s and s[-1][1]>t:q,w=s.pop();b=max(b,(w*(c-q),r-w+1,q,r,c-1));x=q
   s+=(x,t),
 if b[0]<1:return G
 _,e,f,g,j=b;H=j-f>g-e;e+=e>0;g-=H*(g<R-1);f+=1-H;j-=1-H
 for r in I(e,g+1):G[r][f:j+1]=[3]*(j-f+1)
 E=lambda i,L,z,d=0:d or all(O[(v,i)[z]][(i,v)[z]]<1for v in L)
 for i,A,B,P,z in[(r,e,g,[(I(f),f),(I(j+1,C),j<C-1)],1)for r in I(e,g+1)]+[(c,f,j,[(I(e),e),(I(g+1,R),g<R-1)],0)for c in I(f,j+1)]:
  for L,k in P:
   if k*E(i,L,z)*E(i-1,L,z,i==A)*E(i+1,L,z,i==B):
    for v in L:G[(v,i)[z]][(i,v)[z]]=3
 return G
