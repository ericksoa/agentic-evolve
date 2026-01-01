R=range(9)
solve=lambda g:[[g[i][j]or 7*any(8>i+a>-1<j+b<9and g[i+a][j+b]==1for a,b in[(0,1),(0,-1),(1,0),(-1,0)])or 4*any(8>i+a>-1<j+b<9and g[i+a][j+b]==2for a in(-1,1)for b in(-1,1))for j in R]for i in R]
