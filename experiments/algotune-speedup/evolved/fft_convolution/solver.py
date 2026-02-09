"""
Gen0-A: Hybrid dispatch - np.convolve for small signals, np.fft.rfft for larger ones.

Strategy: Use direct convolution (np.convolve) when min(len_x, len_y) < threshold,
otherwise use numpy's real FFT (rfft/irfft) which avoids scipy overhead.
Manual mode handling via full-then-slice to match scipy semantics exactly.
"""

import numpy as np

# Pre-bind for speed
_convolve = np.convolve
_rfft = np.fft.rfft
_irfft = np.fft.irfft
_array = np.asarray

# Threshold: below this min-length, direct convolution wins
_DIRECT_THRESHOLD = 128


class Solver:
    def solve(self, problem: dict) -> dict:
        signal_x = _array(problem["signal_x"], dtype=np.float64)
        signal_y = _array(problem["signal_y"], dtype=np.float64)
        mode = problem.get("mode", "full")

        len_x = len(signal_x)
        len_y = len(signal_y)

        # Handle empty signals
        if len_x == 0 or len_y == 0:
            return {"convolution": np.array([])}

        min_len = min(len_x, len_y)

        if min_len < _DIRECT_THRESHOLD:
            # Direct convolution is faster for small signals
            full_result = _convolve(signal_x, signal_y)
        else:
            # FFT-based convolution using numpy rfft (faster than scipy.signal.fftconvolve)
            full_len = len_x + len_y - 1
            # Power of 2 padding for optimal FFT performance
            fft_size = 1
            while fft_size < full_len:
                fft_size <<= 1

            X = _rfft(signal_x, n=fft_size)
            Y = _rfft(signal_y, n=fft_size)
            full_result = _irfft(X * Y, n=fft_size)[:full_len]

        # Apply mode slicing
        if mode == "full":
            return {"convolution": full_result}

        full_len = len_x + len_y - 1
        if mode == "same":
            start = (full_len - len_x) // 2
            return {"convolution": full_result[start:start + len_x]}
        else:  # valid
            start = min(len_x, len_y) - 1
            valid_len = max(len_x, len_y) - min(len_x, len_y) + 1
            return {"convolution": full_result[start:start + valid_len]}
