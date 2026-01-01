def solve(G):
 I=range;R,C=len(G),len(G[0]);O=sum(G,[])
 if(b:=max([-~(c-a)*-~(k-d),a,d,c,k]for a in I(R)for d in I(C)for c in I(a,R)for k in I(d,C)if~-any(O[r*C+j]for r in I(a,c+1)for j in I(d,k+1)))or[0])[0]<1:return G
 _,e,f,g,j=b;H=j-f>g-e;e+=e>0;g-=H*(g<R-1);f+=1-H;j-=1-H
 for r in I(e,g+1):G[r][f:j+1]=[3]*(j-f+1)
 [G[(v,i)[z]].__setitem__((i,v)[z],3)for a,b,*P,z in((e,g,I(f),I(j+1,C),1),(f,j,I(e),I(g+1,R),0))for i in I(a,b+1)for L in P if L and~-any(O[C*(v,w)[z]+(w,v)[z]]for w in(i,i-(i>a),i+(i<b))for v in L)for v in L]
 return G