import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('gen2x', '.evolve-sdk/optimize_matrix_multiplication/mutations/gen2x.py')
mod = importlib.util.import_module_from_spec(spec)
spec.loader.exec_module(mod)
solver = mod.Solver()
np.random.seed(42)
sizes = [1,2,3,5,10,15,20,25,50,100,200,300,500]
ok_all = True
for n in sizes:
    m = max(n, np.random.randint(n, max(2*n,2)))
    p = max(n, np.random.randint(n, max(2*n,2)))
    A = np.random.randn(n, m).tolist()
    B = np.random.randn(m, p).tolist()
    r = solver.solve({"A": A, "B": B})
    e = np.dot(np.array(A), np.array(B))
    ok = np.allclose(r, e, rtol=1e-5, atol=1e-8)
    ok_all = ok_all and ok
    status = "PASS" if ok else "FAIL"
    print(status, "n="+str(n), "m="+str(m), "p="+str(p))
for n,m,p in [(1,100,1),(100,1,100),(5,200,3),(200,5,200)]:
    A = np.random.randn(n, m).tolist()
    B = np.random.randn(m, p).tolist()
    r = solver.solve({"A": A, "B": B})
    e = np.dot(np.array(A), np.array(B))
    ok = np.allclose(r, e, rtol=1e-5, atol=1e-8)
    ok_all = ok_all and ok
    status = "PASS" if ok else "FAIL"
    print(status, "rect n="+str(n), "m="+str(m), "p="+str(p))
print("All passed:", ok_all)
