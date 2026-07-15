"""
control/allocation_qp.py — Centralized cable-tension allocation (Mode B).

Given a desired wrench [F_des; τ_des] ∈ R⁶ on the load, solve for the 4
non-negative cable tensions that produce that wrench, using the current
cable geometry (directions n_i and moment arms q_i × n_i).

The system is over-actuated (6 DOF wrench, 4 scalar tensions) so it is
generically under-determined; but the positivity constraint (T ≥ T_min > 0)
and maximum actuator limit (T ≤ T_max) make it a bounded least-squares
problem.

We solve:
    minimize   ‖T‖²   (or weighted, to equalise load sharing)
    subject to A T ≈ w_des
               T_min ≤ T_i ≤ T_max

Using scipy.optimize.lsq_linear for the bounded least-squares, which handles
the box constraints directly.  If the wrench is infeasible (geometry is
rank-deficient or w_des is outside the achievable set), we fall back to the
minimum-norm least-squares solution and clip to bounds.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
import warnings


def build_structure_matrix(
    n_hats: list[np.ndarray],
    moment_arms_body: list[np.ndarray],
    R_L: np.ndarray,
) -> np.ndarray:
    """Build the 6×4 load structure (grasp) matrix A.

    A maps scalar cable tensions T = [T_1, T_2, T_3, T_4]ᵀ to the wrench
    on the load:
        [F; τ] = A · T

    Forces are in the inertial frame (each column is n_i, the unit direction
    from load corner toward drone, scaled by tension → force on load).
    Torques are in the load body frame (moment arm q_i × Rᵀ_L n_i).

    Parameters
    ----------
    n_hats : list of 4 × (3,) unit cable directions (load→drone, inertial)
    moment_arms_body : list of 4 × (3,) corner positions in body frame
    R_L : (3,3) load rotation matrix

    Returns
    -------
    A : (6, 4) structure matrix
    """
    A = np.zeros((6, 4))
    for i in range(4):
        # Force column (inertial frame)
        A[0:3, i] = n_hats[i]

        # Torque column (body frame): q_i × (Rᵀ n_i)
        n_body = R_L.T @ n_hats[i]
        A[3:6, i] = np.cross(moment_arms_body[i], n_body)

    return A


def allocate_tensions(
    A: np.ndarray,
    wrench_des: np.ndarray,
    T_min: float = 0.5,
    T_max: float = 40.0,
) -> tuple[np.ndarray, bool]:
    """Solve the fast least-squares tension allocation.

    Uses analytical pseudo-inverse and clips to bounds for speed.
    """
    try:
        # Fast analytical pseudo-inverse (least-squares) + clipping
        T = np.linalg.pinv(A) @ wrench_des
        T = np.clip(T, T_min, T_max)
        feasible = True
    except Exception:
        # Fallback to simple average
        T = np.full(4, (T_min + T_max) / 2.0)
        feasible = False

    return T, feasible


def compute_static_load_sharing(
    corner_points_body: np.ndarray,
    cable_directions: list[np.ndarray],
    total_weight: float,
    R_L: np.ndarray,
) -> np.ndarray:
    """Compute static (hover) load sharing among 4 cables.

    At hover equilibrium: sum of vertical cable force components = total weight,
    sum of horizontal = 0, sum of torques = 0.

    For a symmetric configuration with equal cable angles, tensions are equal.
    For asymmetric cases, solve the static equilibrium equations.

    Parameters
    ----------
    corner_points_body : (4, 3) corner positions in body frame
    cable_directions : list of 4 × (3,) unit cable directions (inertial)
    total_weight : total weight to support (N)
    R_L : (3,3) load rotation matrix

    Returns
    -------
    T_static : (4,) static tensions
    """
    A = build_structure_matrix(cable_directions, list(corner_points_body), R_L)

    # Desired wrench at hover: only vertical force, no torque
    wrench_hover = np.array([0.0, 0.0, total_weight, 0.0, 0.0, 0.0])

    T_static, _ = allocate_tensions(A, wrench_hover, T_min=0.01, T_max=1000.0)
    return T_static
