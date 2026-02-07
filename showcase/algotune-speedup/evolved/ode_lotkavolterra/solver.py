"""
Gen2 Variant B: PI controller + periodic conservation law correction
Mutation: Structural change on parent gen0_f (fitness 742.0)
Parent: gen0_f — DOPRI5 with conservation law correction (end only)

Key change: The parent applies conservation law correction only at the final time.
This mutation:
  1. Adds a PI step-size controller (the champion gen0_d's key advantage)
  2. Applies conservation correction every ~20 accepted steps (not just at the end)
     This keeps the trajectory on the correct energy surface throughout integration,
     preventing error accumulation and allowing more relaxed tolerances.
  3. Relaxes tolerances to 3e-7 since periodic correction recovers accuracy
  4. Uses sqrt-free PI control (from gen2a's insight) to eliminate sqrt from inner loop
  5. max_factor=8.0 for aggressive step growth
"""
import numpy as np
import numba as nb
import math


@nb.njit(cache=True, fastmath=True)
def _compute_hamiltonian(x, y, alpha, beta, delta, gamma):
    """Compute the conserved quantity H = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)."""
    if x <= 0.0 or y <= 0.0:
        return 1e30
    return delta * x - gamma * math.log(x) + beta * y - alpha * math.log(y)


@nb.njit(cache=True, fastmath=True)
def _correct_conservation(x, y, H0, alpha, beta, delta, gamma):
    """Newton's method to project (x,y) onto the level set H(x,y) = H0.
    Lightweight: just 3 iterations for periodic correction (enough to maintain drift < 1e-12)."""
    for _ in range(3):
        if x <= 0.0 or y <= 0.0:
            break
        Hval = delta * x - gamma * math.log(x) + beta * y - alpha * math.log(y)
        err = Hval - H0
        if abs(err) < 1e-13:
            break
        dHdx = delta - gamma / x
        dHdy = beta - alpha / y
        grad_sq = dHdx * dHdx + dHdy * dHdy
        if grad_sq < 1e-30:
            break
        step = err / grad_sq
        x -= step * dHdx
        y -= step * dHdy
        if x <= 0.0:
            x = 1e-15
        if y <= 0.0:
            y = 1e-15
    return x, y


@nb.njit(cache=True, fastmath=True)
def _solve_lv_periodic_correction(t0, t1, x0, y0, alpha, beta, delta, gamma, rtol, atol):
    """DOPRI5 with PI step-size controller + periodic conservation law correction."""

    # Compute initial Hamiltonian for periodic correction
    H0 = _compute_hamiltonian(x0, y0, alpha, beta, delta, gamma)

    a21 = 0.2
    a31 = 0.075
    a32 = 0.225
    a41 = 0.9777777777777777
    a42 = -3.7333333333333334
    a43 = 3.5555555555555554
    a51 = 2.9525986892242035
    a52 = -11.595793324188385
    a53 = 9.822892851699436
    a54 = -0.2908093278463649
    a61 = 2.8462752525252526
    a62 = -10.757575757575758
    a63 = 8.906422717743473
    a64 = 0.2784090909090909
    a65 = -0.2735313036020583

    b1 = 0.09114583333333333
    b3 = 0.44923629829290207
    b4 = 0.6510416666666666
    b5 = -0.322376179245283
    b6 = 0.13095238095238096

    e1 = 0.0012326388888888888
    e3 = -0.0042527702905061394
    e4 = 0.036979166666666666
    e5 = -0.05086379716981132
    e6 = 0.04190476190476190
    e7 = -0.025

    t = t0
    x = x0
    y = y0

    # LV-specific factored RHS
    k1x = x * (alpha - beta * y)
    k1y = y * (delta * x - gamma)

    # Initial step estimate using LV characteristic period
    ag = alpha * gamma
    if ag > 0.0:
        period_est = 6.283185307179586 / math.sqrt(ag)
        h = period_est * 0.01
    else:
        h = 1e-6
    h = min(h, (t1 - t0) * 0.05)
    if h < 1e-10:
        h = 1e-6

    safety = 0.9
    min_factor = 0.2
    max_factor = 8.0

    # PI controller parameters (sqrt-free formulation)
    # Original: factor = safety * err_norm^(-beta1) * prev_err^(beta2)
    # With err_norm = sqrt(err_sq): err_norm^(-beta1) = err_sq^(-beta1/2)
    half_beta1 = 0.7 / 10.0   # 0.07
    half_beta2 = 0.4 / 10.0   # 0.04
    prev_err_sq = 1.0

    # Periodic conservation correction counter
    steps_since_correction = 0
    correction_interval = 20  # Apply correction every 20 accepted steps

    for _ in range(10000000):
        if t >= t1:
            break
        if t + h > t1:
            h = t1 - t
        if h < 1e-14:
            break

        # Stage 2
        xt = x + h * a21 * k1x
        yt = y + h * a21 * k1y
        k2x = xt * (alpha - beta * yt)
        k2y = yt * (delta * xt - gamma)

        # Stage 3
        xt = x + h * (a31 * k1x + a32 * k2x)
        yt = y + h * (a31 * k1y + a32 * k2y)
        k3x = xt * (alpha - beta * yt)
        k3y = yt * (delta * xt - gamma)

        # Stage 4
        xt = x + h * (a41 * k1x + a42 * k2x + a43 * k3x)
        yt = y + h * (a41 * k1y + a42 * k2y + a43 * k3y)
        k4x = xt * (alpha - beta * yt)
        k4y = yt * (delta * xt - gamma)

        # Stage 5
        xt = x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x)
        yt = y + h * (a51 * k1y + a52 * k2y + a53 * k3y + a54 * k4y)
        k5x = xt * (alpha - beta * yt)
        k5y = yt * (delta * xt - gamma)

        # Stage 6
        xt = x + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x)
        yt = y + h * (a61 * k1y + a62 * k2y + a63 * k3y + a64 * k4y + a65 * k5y)
        k6x = xt * (alpha - beta * yt)
        k6y = yt * (delta * xt - gamma)

        # 5th order solution
        xnew = x + h * (b1 * k1x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x)
        ynew = y + h * (b1 * k1y + b3 * k3y + b4 * k4y + b5 * k5y + b6 * k6y)

        # Stage 7 (FSAL)
        k7x = xnew * (alpha - beta * ynew)
        k7y = ynew * (delta * xnew - gamma)

        # Error estimate
        errx = h * (e1 * k1x + e3 * k3x + e4 * k4x + e5 * k5x + e6 * k6x + e7 * k7x)
        erry = h * (e1 * k1y + e3 * k3y + e4 * k4y + e5 * k5y + e6 * k6y + e7 * k7y)

        # Scaled error norm
        sc_x = atol + rtol * max(abs(x), abs(xnew))
        sc_y = atol + rtol * max(abs(y), abs(ynew))
        err_sq = 0.5 * ((errx / sc_x) ** 2 + (erry / sc_y) ** 2)

        if err_sq <= 1.0:
            t += h
            x = xnew
            y = ynew
            k1x = k7x
            k1y = k7y

            # Periodic conservation correction - project back onto energy surface
            steps_since_correction += 1
            if steps_since_correction >= correction_interval:
                steps_since_correction = 0
                if x > 0.0 and y > 0.0:
                    x, y = _correct_conservation(x, y, H0, alpha, beta, delta, gamma)
                    # Recompute k1 after correction since x,y changed
                    k1x = x * (alpha - beta * y)
                    k1y = y * (delta * x - gamma)

            # PI step-size controller WITHOUT sqrt
            if err_sq < 1e-20:
                factor = max_factor
            else:
                factor = safety * (err_sq ** (-half_beta1)) * (prev_err_sq ** half_beta2)
                if factor > max_factor:
                    factor = max_factor
                elif factor < min_factor:
                    factor = min_factor
            prev_err_sq = err_sq
            h *= factor
        else:
            # Standard rejection (sqrt-free)
            factor = safety * (err_sq ** (-0.1))
            if factor < min_factor:
                factor = min_factor
            h *= factor

    # Final conservation correction
    if x > 0.0 and y > 0.0:
        x, y = _correct_conservation(x, y, H0, alpha, beta, delta, gamma)

    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    return x, y


class Solver:
    def __init__(self):
        _solve_lv_periodic_correction(0.0, 1.0, 10.0, 5.0, 1.1, 0.4, 0.1, 0.4, 3e-7, 3e-7)

    def solve(self, problem):
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        delta = float(params["delta"])
        gamma = float(params["gamma"])

        x, y = _solve_lv_periodic_correction(
            t0, t1, float(y0[0]), float(y0[1]),
            alpha, beta, delta, gamma, 3e-7, 3e-7
        )
        return [x, y]
