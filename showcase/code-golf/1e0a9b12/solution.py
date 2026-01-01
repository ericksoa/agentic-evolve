solve=lambda g:[*map(list,zip(*[sorted(c,key=id)for c in zip(*g)]))]
