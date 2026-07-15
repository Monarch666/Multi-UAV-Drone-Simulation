"""
tests/test_single_cable_pendulum.py — Validate cable model against closed-form pendulum.

Test: A single drone held fixed at the origin, with a point mass hanging below
on a single elastic cable.  For small oscillations, the period should match
the ideal pendulum:  T = 2π√(L/g).

The elastic cable model introduces a small deviation (the cable stretches
slightly under load, so the effective length is L0 + δ), but for small
deflections and reasonable stiffness the period should agree within ~2%.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.integrate import solve_ivp


def test_pendulum_period_elastic():
    """Single cable pendulum: compare simulated period vs. analytic 2π√(L/g)."""
    # Parameters
    g = 9.80665
    L0 = 1.2          # cable natural length (m)
    m_load = 1.0      # point mass (kg)
    k_cable = 1500.0   # cable stiffness
    c_cable = 0.5      # low damping to keep oscillation clean
    theta0 = 0.05     # initial deflection angle (rad) — small angle

    # The "drone" is fixed at origin.  The load hangs below on a cable.
    # State: [x, z, vx, vz] of the load (2D for simplicity).
    # Cable attaches at origin (0, 0).

    def rhs(t, y):
        x, z, vx, vz = y
        # Cable vector from load to attachment (origin)
        ex, ez = -x, -z
        L = np.sqrt(ex**2 + ez**2)
        if L < 1e-12:
            return [vx, vz, 0.0, -g]
        nx, nz = ex / L, ez / L
        delta = max(0.0, L - L0)
        v_rel_dot_n = -(vx * nx + vz * nz)
        T = max(0.0, k_cable * delta + c_cable * v_rel_dot_n)
        ax = T * nx / m_load
        az = T * nz / m_load - g
        return [vx, vz, ax, az]

    # Initial state: load displaced by theta0 from vertical
    x0 = L0 * np.sin(theta0)
    z0 = -L0 * np.cos(theta0)
    y0 = [x0, z0, 0.0, 0.0]

    # Integrate for several periods
    T_analytic = 2 * np.pi * np.sqrt(L0 / g)
    T_sim = 8 * T_analytic
    t_eval = np.linspace(0, T_sim, 10000)

    sol = solve_ivp(rhs, [0, T_sim], y0, method='Radau',
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    assert sol.success, f"Integration failed: {sol.message}"

    # Find zero-crossings of x(t) to measure period
    x_arr = sol.y[0]
    crossings = []
    for i in range(1, len(x_arr)):
        if x_arr[i-1] * x_arr[i] < 0 and x_arr[i-1] > 0:  # positive → negative
            # Linear interpolation for crossing time
            frac = x_arr[i-1] / (x_arr[i-1] - x_arr[i])
            t_cross = sol.t[i-1] + frac * (sol.t[i] - sol.t[i-1])
            crossings.append(t_cross)

    assert len(crossings) >= 3, f"Not enough zero crossings: {len(crossings)}"

    # Measured periods
    periods = np.diff(crossings)
    T_measured = np.mean(periods)

    # Account for cable elongation: effective length is slightly longer
    # At rest, cable force = m*g, elongation δ = m*g/k
    delta_static = m_load * g / k_cable
    L_effective = L0 + delta_static
    T_effective = 2 * np.pi * np.sqrt(L_effective / g)

    rel_error_ideal = abs(T_measured - T_analytic) / T_analytic
    rel_error_effective = abs(T_measured - T_effective) / T_effective

    print(f"\nPendulum period test:")
    print(f"  Analytic (ideal):     T = {T_analytic:.6f} s")
    print(f"  Analytic (effective): T = {T_effective:.6f} s")
    print(f"  Measured:             T = {T_measured:.6f} s")
    print(f"  Relative error (ideal):     {rel_error_ideal*100:.4f} %")
    print(f"  Relative error (effective): {rel_error_effective*100:.4f} %")

    # The simulated period should match the effective-length prediction within 2%
    assert rel_error_effective < 0.02, (
        f"Pendulum period error too large: {rel_error_effective*100:.2f}% > 2%"
    )


if __name__ == '__main__':
    test_pendulum_period_elastic()
    print("\n✓ Pendulum period test passed.")
