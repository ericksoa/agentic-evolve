# Gen 1: Basic golfing - shorter variable names, remove comments
def solve(g):
    R,C=len(g),len(g[0])
    m,s,k=set(),set(),0
    for r in range(R):
        for c in range(C):
            if g[r][c]==2:m.add((r,c))
            elif g[r][c]:s.add((r,c));k=g[r][c]
    mr={r for r,c in m}
    mc={c for r,c in m}
    sr={r for r,c in s}
    sc={c for r,c in s}
    o=[[3]*C for _ in range(R)]
    for r,c in s:o[r][c]=k
    for r,c in m:o[r][c]=k
    if len(mr)==1:
        M=list(mr)[0]
        e=M+.5 if min(sr)>M else M-.5
        for r,c in s:
            n=int(2*e-r)
            if 0<=n<R:o[n][c]=k
    elif len(mc)==1:
        M=list(mc)[0]
        e=M-.5 if max(sc)<M else M+.5
        for r,c in s:
            n=int(2*e-c)
            if 0<=n<C:o[r][n]=k
    return o
