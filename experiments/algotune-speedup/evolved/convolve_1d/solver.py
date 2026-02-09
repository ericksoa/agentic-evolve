"""
Gen1c: Drop Numba, use scipy.fft with next_fast_len + hybrid dispatch
Mutation: Algorithm swap - replace Numba JIT with direct scipy.fft calls
Parent: gen0_d (Numba JIT direct convolution with parallelization)

Key changes from parent:
- Removes Numba dependency entirely (JIT compilation can fail/timeout in
  constrained environments, and import alone costs ~1s)
- Uses scipy.fft.rfft/irfft (pocketfft backend, faster than numpy.fft)
- Uses next_fast_len for optimal FFT padding (products of small primes,
  much better than power-of-2 for non-power-of-2 output lengths)
- Uses np.convolve (C-optimized) for small inputs where FFT overhead dominates
- In-place frequency-domain multiply to reduce allocations
- Pre-binds all functions to local variables for fast dispatch
- Caches FFT length computations for repeated problem sizes

Hypothesis: Numba JIT warmup cost + potential compilation failures make it
unreliable. scipy.fft with next_fast_len is universally fast, avoids JIT
compilation entirely, and uses the superior pocketfft backend. For small
arrays, np.convolve's C implementation is faster than any FFT approach.
Calling rfft/irfft directly avoids fftconvolve's input validation overhead.
"""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len

# Pre-bind to avoid attribute lookups in hot path
_convolve = np.convolve
_rfft = rfft
_irfft = irfft
_next_fast_len = next_fast_len


class Solver:
    def __init__(self):
        # Cache FFT length computations
        self._len_cache = {}

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        na = len(a)
        nb = len(b)

        if na * nb <= 40000:
            # Direct convolution via numpy's C implementation
            # Faster for small arrays - avoids FFT plan overhead
            return _convolve(a, b, mode='full')

        out_len = na + nb - 1

        # Cache FFT length lookup
        cache = self._len_cache
        if out_len in cache:
            fft_len = cache[out_len]
        else:
            fft_len = _next_fast_len(out_len)
            cache[out_len] = fft_len

        # FFT-based convolution: scipy.fft uses pocketfft (fast)
        fa = _rfft(a, n=fft_len)
        fa *= _rfft(b, n=fft_len)  # in-place multiply saves one allocation
        return _irfft(fa, n=fft_len)[:out_len]
