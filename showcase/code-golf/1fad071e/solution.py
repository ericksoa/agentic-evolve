def solve(g):R=range(8);c=sum(g[i][j:j+2]+g[i+1][j:j+2]==[1]*4for i in R for j in R);return[[1]*c+[0]*(5-c)]
