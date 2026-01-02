R=range(11)
D=-1,0,1
solve=lambda g:[[max(0,*[g[r+a][c+b]for r in R for c in R if g[r][c]==5<11>r+a>~0<c+b])for b in D]for a in D]