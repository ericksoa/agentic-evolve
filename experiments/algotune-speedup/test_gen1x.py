import numpy as np
import importlib.util
from scipy.signal import fftconvolve

spec = importlib.util.spec_from_file_location(
    'gen1x',
    '/Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/algotune-speedup/.evolve-sdk/optimize_fft_convolution_solve/mutations/gen1x.py'
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
solver = mod.Solver()
print('Solver loaded.')

sizes = [
    (5, 3), (1, 10), (10, 1), (50, 30), (200, 150),
    (500, 300), (1000, 800), (3, 3), (128, 128), (129, 129), (1, 1),
]
modes = ['full', 'same', 'valid']
rng = np.random.RandomState(42)

total = 0
passed = 0
failed = 0
errors = []
TOLERANCE = 1e-8

print()
print('--- Signal convolution tests ---')
for (lx, ly) in sizes:
    signal_x = rng.randn(lx)
    signal_y = rng.randn(ly)
    for mode in modes:
        total += 1
        label = '(%d,%d) %s' % (lx, ly, mode)
        try:
            ref = fftconvolve(signal_x, signal_y, mode=mode)
            result = solver.solve({
                'signal_x': signal_x.tolist(),
                'signal_y': signal_y.tolist(),
                'mode': mode,
            })
            out = np.asarray(result['convolution'])
            if ref.shape != out.shape:
                msg = 'FAIL %s: shape mismatch ref=%s got=%s' % (label, ref.shape, out.shape)
                errors.append(msg)
                failed += 1
                print('  ' + msg)
                continue
            max_err = np.max(np.abs(ref - out))
            if max_err >= TOLERANCE:
                msg = 'FAIL %s: max_abs_error=%.3e exceeds 1e-8' % (label, max_err)
                errors.append(msg)
                failed += 1
                print('  ' + msg)
                continue
            passed += 1
            print('  PASS %s (max_err=%.2e)' % (label, max_err))
        except Exception as e:
            msg = 'ERROR %s: %s' % (label, e)
            errors.append(msg)
            failed += 1
            print('  ' + msg)

print()
print('--- Edge cases: empty signals ---')
edge_cases = [
    ('empty x', [], [1.0, 2.0]),
    ('empty y', [1.0, 2.0], []),
    ('both empty', [], []),
]
for desc, sx, sy in edge_cases:
    total += 1
    try:
        result = solver.solve({'signal_x': sx, 'signal_y': sy, 'mode': 'full'})
        out = np.asarray(result['convolution'])
        if out.size == 0:
            passed += 1
            print('  PASS edge (%s): returned empty array' % desc)
        else:
            msg = 'FAIL edge (%s): expected empty, got size=%d' % (desc, out.size)
            errors.append(msg)
            failed += 1
            print('  ' + msg)
    except Exception as e:
        msg = 'ERROR edge (%s): %s' % (desc, e)
        errors.append(msg)
        failed += 1
        print('  ' + msg)

print()
print('=' * 60)
print('RESULTS: %d/%d passed, %d failed' % (passed, total, failed))
print('=' * 60)
if errors:
    print()
    print('Failures:')
    for e in errors:
        print('  ' + e)
else:
    print('All tests passed.')
