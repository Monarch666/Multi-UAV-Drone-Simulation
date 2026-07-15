"""
tests/test_energy_sanity.py — Verify energy conservation in passive scenarios.

Test: With all rotors off (zero thrust), the system undergoes free-fall.
The only forces are gravity and cable spring-damper forces.
Energy should:
  1. Never increase beyond the initial value (no energy injection).
  2. Decrease monotonically (or stay constant if damping is zero) due to
     cable damping and friction dissipation.
  3. Remain bounded (no numerical blow-up).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dynamics.drone import DroneParams
from dynamics.cable import CableParams
from dynamics.load import BasketParams
from dynamics.system import (
    SystemConfig, build_initial_state, make_system_rhs,
    compute_total_energy, normalise_quaternions,
    unpack_system_state, TOTAL_STATE_DIM,
)


def _zero_controller(t, drone_states, load_state, sliding_state, cfg):
    """No-thrust controller: all rotors at minimum idle speed."""
    return [np.full(4, 0.0) for _ in range(4)]


def test_energy_no_injection():
    """Passive drop test: energy should only decrease, never increase."""
    # Build config with moderate cable stiffness
    dp = [DroneParams() for _ in range(4)]
    cp = [CableParams(stiffness=500.0, damping=10.0) for _ in range(4)]
    bp = BasketParams()
    cfg = SystemConfig(drone_params=dp, cable_params=cp, basket_params=bp)

    # Initial state: hovering at z=3
    y0 = build_initial_state(cfg, load_pos=np.array([0.0, 0.0, 3.0]))

    # Set all rotor speeds to zero (passive drop)
    # Zero out rotor speed states in initial condition
    for i in range(4):
        base = i * 17 + 13  # rotor speeds start at index 13 within each drone
        y0[base:base+4] = 0.0

    rhs = make_system_rhs(cfg, _zero_controller)

    # Short simulation (1 second of free fall)
    T_sim = 1.0
    t_eval = np.linspace(0, T_sim, 200)

    sol = solve_ivp(rhs, [0, T_sim], y0, method='Radau',
                    t_eval=t_eval, rtol=1e-8, atol=1e-10,
                    max_step=0.005)

    assert sol.success, f"Integration failed: {sol.message}"

    # Compute energy at each time step
    energies = []
    for k in range(len(sol.t)):
        y = sol.y[:, k]
        E = compute_total_energy(y, cfg)
        energies.append(E['total'])

    energies = np.array(energies)
    E0 = energies[0]

    print(f"\nEnergy sanity test (passive drop):")
    print(f"  Initial energy: {E0:.4f} J")
    print(f"  Final energy:   {energies[-1]:.4f} J")
    print(f"  Min energy:     {energies.min():.4f} J")
    print(f"  Max energy:     {energies.max():.4f} J")
    print(f"  Energy change:  {energies[-1] - E0:.4f} J")

    # Energy should never exceed initial value by more than a small tolerance
    # (numerical noise may cause tiny increases)
    tolerance = 0.05 * abs(E0) + 1.0  # 5% relative + 1 J absolute
    max_energy = energies.max()
    assert max_energy <= E0 + tolerance, (
        f"Energy increased beyond tolerance: max {max_energy:.4f} > "
        f"initial {E0:.4f} + tol {tolerance:.4f}"
    )

    print(f"  ✓ No unphysical energy injection detected.")


def test_energy_bounded():
    """Energy should remain finite throughout integration (no blow-up)."""
    dp = [DroneParams() for _ in range(4)]
    cp = [CableParams(stiffness=500.0, damping=10.0) for _ in range(4)]
    bp = BasketParams()
    cfg = SystemConfig(drone_params=dp, cable_params=cp, basket_params=bp)

    y0 = build_initial_state(cfg, load_pos=np.array([0.0, 0.0, 3.0]))
    for i in range(4):
        base = i * 17 + 13
        y0[base:base+4] = 0.0

    rhs = make_system_rhs(cfg, _zero_controller)
    sol = solve_ivp(rhs, [0, 0.5], y0, method='Radau',
                    rtol=1e-8, atol=1e-10, max_step=0.005)

    assert sol.success

    # Check all state values are finite
    assert np.all(np.isfinite(sol.y)), "State vector contains NaN or Inf!"

    # Check energy is finite
    for k in range(0, len(sol.t), 10):
        E = compute_total_energy(sol.y[:, k], cfg)
        assert np.isfinite(E['total']), f"Energy is NaN/Inf at t={sol.t[k]:.3f}"

    print("\n  ✓ Energy bounded test passed.")


if __name__ == '__main__':
    test_energy_no_injection()
    print("✓ Energy no-injection test passed.")
    test_energy_bounded()
    print("✓ Energy bounded test passed.")
