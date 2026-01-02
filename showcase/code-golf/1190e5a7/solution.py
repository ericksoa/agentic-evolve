def solve(g):
 d=next(r[0]for r in g if len({*r})<2);t=[*zip(*g)]
 return[[[v for r in g for v in r if v-d][0]]*(sum(len({*c})<2for c in t)+1)for _ in range(sum(len({*r})<2for r in g)+1)]
