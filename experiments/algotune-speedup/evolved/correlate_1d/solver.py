"""
Gen1a: Custom smooth-number FFT sizing (no scipy overhead).
Mutation of gen0_d: Replace power-of-2 padding with a precomputed lookup table
of 5-smooth numbers (products of 2,3,5). Numpy's pocketfft handles these efficiently.
On average, smooth-number padding gives ~1.4x shorter FFT sizes than power-of-2,
meaning less computation while keeping numpy.fft's low overhead.
Also adds tiny-array tier (np.correlate) and caches append for reduced dispatch overhead.
"""
import numpy as np
from numpy.fft import rfft, irfft

_np_rfft = rfft
_np_irfft = irfft
_np_convolve = np.convolve
_np_correlate = np.correlate

# Precompute all 5-smooth numbers up to 2^20 (~1M) for fast FFT sizing
# 5-smooth = numbers of the form 2^a * 3^b * 5^c
def _build_smooth_table():
    limit = 1 << 20  # ~1M, covers all practical FFT sizes
    smooth = []
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                smooth.append(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    smooth.sort()
    return smooth

_SMOOTH = _build_smooth_table()

def _next_smooth(n):
    """Find smallest 5-smooth number >= n using binary search."""
    lo, hi = 0, len(_SMOOTH) - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if _SMOOTH[mid] < n:
            lo = mid + 1
        else:
            hi = mid
    return _SMOOTH[lo]


class Solver:
    def solve(self, problem: list) -> list:
        results = []
        append = results.append

        for a, b in problem:
            na = len(a)
            nb = len(b)
            prod = na * nb

            if prod <= 500:
                # Tiny: direct np.correlate has minimal overhead
                append(_np_correlate(a, b, mode='full'))
            elif prod <= 30000:
                # Small: direct convolution with reversed b
                append(_np_convolve(a, b[::-1], mode='full'))
            else:
                # Large: FFT-based correlation via convolve(a, b[::-1])
                out_len = na + nb - 1
                fft_len = _next_smooth(out_len)

                b_rev = b[::-1]
                fa = _np_rfft(a, n=fft_len)
                fa *= _np_rfft(b_rev, n=fft_len)
                append(_np_irfft(fa, n=fft_len)[:out_len])

        return results
