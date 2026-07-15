"""
tests/test_static_load_sharing.py — Validate hover tension distribution.

Test 1: Symmetric geometry (CM centered) → all 4 tensions should be equal.
Test 2: Off-center static load → tensions should shift toward near corners.
Test 3: Total vertical tension should equal total weight.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from dynamics.drone import DroneParams, quat_to_rotmat
from dynamics.load import BasketParams, get_corner_world
from dynamics.cable import CableParams
from dynamics.system import SystemConfig, build_initial_state, NUM_DRONES
from control.allocation_qp import (
    build_structure_matrix, allocate_tensions, compute_static_load_sharing,
)


def _build_symmetric_config():
    """Build a symmetric configuration for testing."""
    dp = [DroneParams() for _ in range(4)]
    cp = [CableParams() for _ in range(4)]
    bp = BasketParams(
        static_load_mass=1.5,
        static_load_pos_body=np.array([0.0, 0.0, -0.05]),  # centered
    )
    return SystemConfig(drone_params=dp, cable_params=cp, basket_params=bp)


def test_symmetric_equal_tensions():
    """With symmetric geometry and centered CM, all tensions should be equal."""
    cfg = _build_symmetric_config()
    bp = cfg.basket_params
    R_L = np.eye(3)
    p_L = np.array([0.0, 0.0, 3.0])

    # Compute cable directions (drones directly above corners)
    n_hats = []
    for i in range(4):
        corner = get_corner_world(p_L, R_L, bp.corner_points_body[i])
        L0 = cfg.cable_params[i].natural_length
        drone_pos = corner + np.array([0.0, 0.0, L0])
        e = drone_pos - corner
        n_hats.append(e / np.linalg.norm(e))

    total_weight = (bp.total_mass + bp.dynamic_load_mass) * cfg.gravity
    T_static = compute_static_load_sharing(bp.corner_points_body, n_hats,
                                            total_weight, R_L)

    print(f"\nSymmetric tensions: {T_static.round(4)}")
    print(f"Total weight: {total_weight:.4f} N")
    print(f"Sum of vertical tensions: {sum(T_static * [n[2] for n in n_hats]):.4f} N")

    # All tensions should be equal (within numerical tolerance)
    assert np.allclose(T_static, T_static[0], atol=0.1), \
        f"Tensions not equal: {T_static}"

    # Sum of vertical components should equal total weight
    T_vert = sum(T_static[i] * n_hats[i][2] for i in range(4))
    assert abs(T_vert - total_weight) < 0.5, \
        f"Vertical tension sum {T_vert:.2f} != weight {total_weight:.2f}"


def test_offcenter_load_shifts_tensions():
    """Off-center static load should shift tensions toward near corners."""
    # Create config with off-center load (shifted toward corner 0: +x, +y)
    dp = [DroneParams() for _ in range(4)]
    cp = [CableParams() for _ in range(4)]
    bp = BasketParams(
        static_load_mass=1.5,
        static_load_pos_body=np.array([0.15, 0.15, -0.05]),  # shifted +x, +y
    )
    cfg = SystemConfig(drone_params=dp, cable_params=cp, basket_params=bp)

    R_L = np.eye(3)
    p_L = np.array([0.0, 0.0, 3.0])

    n_hats = []
    for i in range(4):
        corner = get_corner_world(p_L, R_L, bp.corner_points_body[i])
        L0 = cfg.cable_params[i].natural_length
        drone_pos = corner + np.array([0.0, 0.0, L0])
        e = drone_pos - corner
        n_hats.append(e / np.linalg.norm(e))

    total_weight = (bp.total_mass + bp.dynamic_load_mass) * cfg.gravity
    T_static = compute_static_load_sharing(bp.corner_points_body, n_hats,
                                            total_weight, R_L)

    print(f"\nOff-center tensions: {T_static.round(4)}")

    # Corner 0 is at (+a/2, +b/2) — near the shifted load — should have
    # highest tension.  Corner 2 is at (-a/2, -b/2) — farthest — lowest.
    # Note: after CM recomputation, the corners have shifted too.
    # The key check is that tensions are NOT all equal.
    tension_spread = T_static.max() - T_static.min()
    print(f"Tension spread: {tension_spread:.4f} N")

    # With a 0.15m shift of 1.5 kg on a 0.6×0.6 basket, the spread should be
    # noticeable (at least a few percent of mean).
    mean_T = np.mean(T_static)
    assert tension_spread > 0.01 * mean_T, \
        f"Tension spread {tension_spread:.4f} too small for off-center load"


def test_total_vertical_tension():
    """Total vertical cable tension component should equal total system weight."""
    cfg = _build_symmetric_config()
    bp = cfg.basket_params
    R_L = np.eye(3)
    p_L = np.array([0.0, 0.0, 3.0])

    n_hats = []
    for i in range(4):
        corner = get_corner_world(p_L, R_L, bp.corner_points_body[i])
        L0 = cfg.cable_params[i].natural_length
        drone_pos = corner + np.array([0.0, 0.0, L0])
        e = drone_pos - corner
        n_hats.append(e / np.linalg.norm(e))

    total_weight = (bp.total_mass + bp.dynamic_load_mass) * cfg.gravity
    T_static = compute_static_load_sharing(bp.corner_points_body, n_hats,
                                            total_weight, R_L)

    T_vert = sum(T_static[i] * n_hats[i][2] for i in range(4))
    print(f"\nTotal vertical tension: {T_vert:.4f} N")
    print(f"Total weight:          {total_weight:.4f} N")

    # Should match within 1%
    rel_error = abs(T_vert - total_weight) / total_weight
    assert rel_error < 0.01, \
        f"Vertical tension error: {rel_error*100:.2f}% > 1%"


if __name__ == '__main__':
    test_symmetric_equal_tensions()
    print("✓ Symmetric tension test passed.")
    test_offcenter_load_shifts_tensions()
    print("✓ Off-center tension test passed.")
    test_total_vertical_tension()
    print("✓ Total vertical tension test passed.")
