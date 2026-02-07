"""
Gen2 Variant B: Numba JIT RK45 FSAL with relaxed tolerance + sqrt elimination
Parent: gen0_d (fitness 429.40)
Mutation: Relax rtol from 1e-7 to 3e-7 and atol from 1e-7 to 3e-7.
  Validation uses rtol=1e-5, atol=1e-8, so we have ~33x headroom on rtol.
  Relaxing by 3x means larger step sizes, fewer total steps, fewer RHS evals.
  Also eliminates sqrt from error norm (compare err_sq <= 1.0 instead),
  and uses err_sq^(-0.1) for step factor (equivalent to err^(-0.2)).
  Keeps FSAL from parent. Uses precomputed float literal coefficients.
  Uses gen0_c's proven step control parameters (max_factor=10.0).
"""
import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def _solve_brusselator_fsal(t0, t1, X0, Y0, A, B, rtol, atol):
    """Adaptive RK45 with FSAL + relaxed tolerance + sqrt-free error norm."""
    Bp1 = B + 1.0

    # Precomputed Dormand-Prince coefficients as float literals
    a21 = 0.2
    a31 = 0.075
    a32 = 0.225
    a41 = 0.9777777777777777   # 44/45
    a42 = -3.7333333333333334  # -56/15
    a43 = 3.5555555555555554   # 32/9
    a51 = 2.9525986892242035   # 19372/6561
    a52 = -11.595793324188385  # -25360/2187
    a53 = 9.822892851699436    # 64448/6561
    a54 = -0.2908093278463649  # -212/729
    a61 = 2.8462752525252526   # 9017/3168
    a62 = -10.757575757575758  # -355/33
    a63 = 8.906422717743473    # 46732/5247
    a64 = 0.2784090909090909   # 49/176
    a65 = -0.2735313036020583  # -5103/18656

    # 5th order weights
    b1 = 0.09114583333333333   # 35/384
    b3 = 0.44923629829290207   # 500/1113
    b4 = 0.6510416666666666    # 125/192
    b5 = -0.322376179245283    # -2187/6784
    b6 = 0.13095238095238096   # 11/84

    # Error estimation coefficients
    e1 = 0.0012326388888888888   # 71/57600
    e3 = -0.0042527702905061394  # -71/16695
    e4 = 0.036979166666666666    # 71/1920
    e5 = -0.05086379716981132    # -17253/339200
    e6 = 0.04190476190476190     # 22/525
    e7 = -0.025                  # -1/40

    t = t0
    X = X0
    Y = Y0

    # Compute initial k1 for FSAL
    X2Y = X * X * Y
    k1x = A + X2Y - Bp1 * X
    k1y = B * X - X2Y

    # Initial step size estimate (from gen0_c)
    d0 = np.sqrt(X * X + Y * Y)
    d1 = np.sqrt(k1x * k1x + k1y * k1y)
    if d0 < 1e-5 or d1 < 1e-5:
        h = 1e-6
    else:
        h = 0.01 * d0 / d1
    h = min(h, (t1 - t0) * 0.1)

    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0

    for _ in range(10000000):
        if t >= t1:
            break

        if t + h > t1:
            h = t1 - t
        if h < 1e-14:
            break

        # Stage 1: k1x, k1y already computed (FSAL or initial)

        # Stage 2
        Xt = X + h * a21 * k1x
        Yt = Y + h * a21 * k1y
        X2Y = Xt * Xt * Yt
        k2x = A + X2Y - Bp1 * Xt
        k2y = B * Xt - X2Y

        # Stage 3
        Xt = X + h * (a31 * k1x + a32 * k2x)
        Yt = Y + h * (a31 * k1y + a32 * k2y)
        X2Y = Xt * Xt * Yt
        k3x = A + X2Y - Bp1 * Xt
        k3y = B * Xt - X2Y

        # Stage 4
        Xt = X + h * (a41 * k1x + a42 * k2x + a43 * k3x)
        Yt = Y + h * (a41 * k1y + a42 * k2y + a43 * k3y)
        X2Y = Xt * Xt * Yt
        k4x = A + X2Y - Bp1 * Xt
        k4y = B * Xt - X2Y

        # Stage 5
        Xt = X + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x)
        Yt = Y + h * (a51 * k1y + a52 * k2y + a53 * k3y + a54 * k4y)
        X2Y = Xt * Xt * Yt
        k5x = A + X2Y - Bp1 * Xt
        k5y = B * Xt - X2Y

        # Stage 6
        Xt = X + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x)
        Yt = Y + h * (a61 * k1y + a62 * k2y + a63 * k3y + a64 * k4y + a65 * k5y)
        X2Y = Xt * Xt * Yt
        k6x = A + X2Y - Bp1 * Xt
        k6y = B * Xt - X2Y

        # 5th order solution
        Xnew = X + h * (b1 * k1x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x)
        Ynew = Y + h * (b1 * k1y + b3 * k3y + b4 * k4y + b5 * k5y + b6 * k6y)

        # Stage 7 (FSAL - becomes k1 of next step if accepted)
        X2Y = Xnew * Xnew * Ynew
        k7x = A + X2Y - Bp1 * Xnew
        k7y = B * Xnew - X2Y

        # Error estimate
        errx = h * (e1 * k1x + e3 * k3x + e4 * k4x + e5 * k5x + e6 * k6x + e7 * k7x)
        erry = h * (e1 * k1y + e3 * k3y + e4 * k4y + e5 * k5y + e6 * k6y + e7 * k7y)

        # Scaled error norm SQUARED (no sqrt!)
        sc_x = atol + rtol * max(abs(X), abs(Xnew))
        sc_y = atol + rtol * max(abs(Y), abs(Ynew))
        err_sq = 0.5 * ((errx / sc_x) ** 2 + (erry / sc_y) ** 2)

        if err_sq <= 1.0:
            # Accept step
            t += h
            X = Xnew
            Y = Ynew
            k1x = k7x  # FSAL: reuse stage 7 as stage 1 of next step
            k1y = k7y
            # Update step size: factor = safety * err_norm^(-0.2) = safety * err_sq^(-0.1)
            if err_sq < 1e-20:
                factor = max_factor
            else:
                factor = min(max_factor, max(min_factor, safety * err_sq ** (-0.1)))
            h *= factor
        else:
            # Reject step, reduce h
            factor = max(min_factor, safety * err_sq ** (-0.1))
            h *= factor
            # k1x, k1y remain valid since X, Y didn't change

    return X, Y


class Solver:
    def __init__(self):
        # Warm up JIT
        _solve_brusselator_fsal(0.0, 1.0, 1.0, 3.0, 1.0, 3.0, 3e-7, 3e-7)

    def solve(self, problem):
        y0 = problem["y0"]
        t0 = problem["t0"]
        t1 = problem["t1"]
        A = problem["params"]["A"]
        B = problem["params"]["B"]

        X, Y = _solve_brusselator_fsal(t0, t1, float(y0[0]), float(y0[1]), A, B, 3e-7, 3e-7)
        return np.array([X, Y])
