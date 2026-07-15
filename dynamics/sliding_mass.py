"""
dynamics/sliding_mass.py — Dynamic (sliding) load on the basket floor.

The sliding mass m_d is constrained to move on the basket's floor plane
(a 2-D plane fixed in the load body frame at height h_floor).

Because the floor is part of a translating AND rotating rigid body, the
sliding mass experiences fictitious forces (Coriolis, centrifugal,
Euler-force) in addition to gravity, friction, and the floor normal force.
These are the physically interesting coupling effects that make the dynamic
load non-trivial.

Relative coordinates
--------------------
s = (s_x, s_y)      — 2-D position of m_d on the floor, in body-frame coords
ṡ = (ṡ_x, ṡ_y)     — 2-D velocity relative to the floor
ρ = (s_x, s_y, h_floor)  — 3-D body-frame position of m_d relative to load CM

World position: p_d = p_L + R_L · ρ

Friction model
--------------
We use a smooth regularised friction model to avoid discontinuities that would
cause integrator chattering:

    F_fric = -μ_eff(‖ṡ‖) · N_z · ṡ / √(‖ṡ‖² + ε²)  −  c_visc · ṡ

where N_z is the normal force magnitude and:
    μ_eff(v) = μ_k + (μ_s − μ_k) · exp(−(v/v_scale)²)

This transitions smoothly from static to kinetic friction without explicit
stick-slip event detection, which is important for adaptive-step integrators.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# State packing
# ---------------------------------------------------------------------------

SLIDING_MASS_STATE_DIM = 4  # s_x, s_y, sdot_x, sdot_y


def pack_sliding_state(s: np.ndarray, sdot: np.ndarray) -> np.ndarray:
    """Pack sliding mass state into flat array (4 elements)."""
    return np.concatenate([s, sdot])


def unpack_sliding_state(x: np.ndarray):
    """Unpack flat array into (s, sdot), each (2,)."""
    return x[0:2], x[2:4]


# ---------------------------------------------------------------------------
# Projection matrix: 3-D body frame → 2-D floor plane
# ---------------------------------------------------------------------------

# P maps 3-D body-frame vector to 2-D floor coords: v_2d = P @ v_3d
# P^T maps 2-D floor coords back to 3-D body frame (with z=0):
#   v_3d_floor = P^T @ v_2d  ⟹  (v_x, v_y, 0)
P_FLOOR = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=float)


# ---------------------------------------------------------------------------
# Friction force
# ---------------------------------------------------------------------------

def smooth_friction_force(
    sdot: np.ndarray,
    normal_force_mag: float,
    mu_k: float,
    mu_s: float,
    visc_damping: float,
    v_scale: float = 0.01,
) -> np.ndarray:
    """Compute regularised 2-D friction force on the sliding mass.

    Parameters
    ----------
    sdot : (2,) relative velocity on floor
    normal_force_mag : scalar N_z (positive = pushes mass into floor)
    mu_k, mu_s : kinetic and static friction coefficients
    visc_damping : viscous damping coefficient (Ns/m)
    v_scale : regularisation velocity (m/s)

    Returns
    -------
    F_fric : (2,) friction force in floor-plane coordinates
    """
    speed = np.linalg.norm(sdot)
    eps = 1e-10

    # Effective friction coefficient (smooth transition)
    mu_eff = mu_k + (mu_s - mu_k) * np.exp(-(speed / v_scale) ** 2)

    # Direction (regularised to avoid division by zero and stiffness)
    direction = sdot / np.sqrt(speed**2 + v_scale**2)

    # Coulomb-type + viscous
    F_fric = -mu_eff * abs(normal_force_mag) * direction - visc_damping * sdot

    return F_fric


# ---------------------------------------------------------------------------
# Sliding mass dynamics in the rotating body frame
# ---------------------------------------------------------------------------

def sliding_mass_derivatives(
    s: np.ndarray,
    sdot: np.ndarray,
    omega_L_body: np.ndarray,
    omega_L_dot_body: np.ndarray,
    a_L_inertial: np.ndarray,
    R_L: np.ndarray,
    floor_height: float,
    m_d: float,
    mu_k: float,
    mu_s: float,
    visc_damping: float,
    gravity: float,
    v_scale: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute acceleration of the sliding mass in body-frame floor coords.

    This implements the full rotating-frame kinematics including Coriolis,
    centrifugal, and Euler (angular acceleration) pseudo-forces.

    The absolute acceleration of the sliding mass (in the inertial frame) is:
        a_d = a_L + R_L (ω̇_L × ρ + ω_L × (ω_L × ρ) + 2 ω_L × P^T ṡ + P^T s̈)

    We solve for s̈ (the 2-D relative acceleration on the floor) by projecting
    into the floor plane and applying the forces acting on m_d.

    Parameters
    ----------
    s        : (2,) position on floor [s_x, s_y]
    sdot     : (2,) velocity on floor [ṡ_x, ṡ_y]
    omega_L_body     : (3,) angular velocity of load in body frame
    omega_L_dot_body : (3,) angular acceleration of load in body frame
    a_L_inertial     : (3,) linear acceleration of load CM in inertial frame
    R_L      : (3,3) rotation matrix load body → inertial
    floor_height : scalar h_floor
    m_d      : mass of sliding load (kg)
    mu_k, mu_s : friction coefficients
    visc_damping : viscous damping coefficient
    gravity  : gravitational acceleration
    v_scale  : friction regularisation velocity

    Returns
    -------
    sddot    : (2,) acceleration of sliding mass on floor
    F_normal : (3,) normal force vector in body frame (z-component)
    F_friction_body : (3,) friction force in body frame (for coupling back to load)
    """
    # 3-D position of m_d relative to load CM in body frame
    rho = np.array([s[0], s[1], floor_height])

    # Velocity of m_d relative to load CM in body frame (floor plane only)
    sdot_3d = np.array([sdot[0], sdot[1], 0.0])

    # ---- Fictitious accelerations in body frame ----
    # Euler term: ω̇_L × ρ
    a_euler = np.cross(omega_L_dot_body, rho)

    # Centrifugal: ω_L × (ω_L × ρ)
    a_centrifugal = np.cross(omega_L_body, np.cross(omega_L_body, rho))

    # Coriolis: 2 ω_L × ṡ_body (body-frame relative velocity)
    a_coriolis = 2.0 * np.cross(omega_L_body, sdot_3d)

    # Transport acceleration of load CM expressed in body frame
    a_L_body = R_L.T @ a_L_inertial

    # ---- Gravity in body frame ----
    g_body = R_L.T @ np.array([0.0, 0.0, -gravity])

    # ---- Total "pseudo" acceleration that m_d sees in the body frame ----
    # Newton's second law in the non-inertial frame:
    #   m_d s̈_body = m_d g_body − m_d (a_L_body + a_euler + a_centrifugal + a_coriolis)
    #                + F_normal + F_friction
    #
    # The normal force constrains the mass to the floor (z-component = 0).
    # We solve for the floor-plane (x,y) components of s̈ and the z-component
    # of the normal force simultaneously.

    # Effective specific force (acceleration) driving the mass, WITHOUT
    # normal force and friction, in body frame
    a_eff_body = g_body - a_L_body - a_euler - a_centrifugal - a_coriolis

    # --- Normal force ---
    # The floor constrains z-acceleration to zero (mass stays on floor).
    # In the z-direction:  0 = a_eff_z + N_z / m_d
    # → N_z = -m_d * a_eff_z
    N_z = -m_d * a_eff_body[2]

    # Ensure N_z is non-negative (mass rests ON floor, not pulled through it)
    # If N_z < 0 the mass would lift off — physically this means the constraint
    # is inactive.  For simplicity we clamp to zero (mass stays on floor plane).
    N_z = max(0.0, N_z)

    # --- Friction ---
    F_fric_2d = smooth_friction_force(
        sdot, N_z, mu_k, mu_s, visc_damping, v_scale
    )

    # --- Floor-plane acceleration ---
    # m_d s̈_2d = m_d * a_eff_body_2d + F_fric_2d
    a_eff_2d = P_FLOOR @ a_eff_body   # project to 2-D
    sddot = a_eff_2d + F_fric_2d / m_d

    # ---- Forces for coupling back to load (in body frame) ----
    F_normal_body = np.array([0.0, 0.0, N_z])
    F_friction_body = np.array([F_fric_2d[0], F_fric_2d[1], 0.0])

    return sddot, F_normal_body, F_friction_body


def sliding_mass_reaction_on_load(
    s: np.ndarray,
    sddot: np.ndarray,
    F_normal_body: np.ndarray,
    F_friction_body: np.ndarray,
    floor_height: float,
    m_d: float,
    R_L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reaction force and torque from sliding mass on the load.

    By Newton's third law, the load experiences the negative of all forces
    exerted on the sliding mass by the load (normal + friction), PLUS the
    inertial reaction from accelerating the mass.

    However, since we are solving the coupled system, the simplest correct
    approach is to compute the force/torque that m_d exerts on the load:

    F_reaction_inertial = -R_L @ (F_normal_body + F_friction_body)
        + m_d * [inertial acceleration terms already in the load EOM coupling]

    For the decoupled formulation (where load EOM is written for m_L only),
    we return the constraint forces:

    Returns
    -------
    F_reaction_inertial : (3,) — force on load CM in inertial frame
    tau_reaction_body   : (3,) — torque on load about CM in body frame
    """
    rho = np.array([s[0], s[1], floor_height])

    # Reaction forces (Newton's 3rd law) in body frame
    F_react_body = -(F_normal_body + F_friction_body)

    # Also include the inertial reaction: -m_d * s̈ expressed in body frame
    sddot_3d = np.array([sddot[0], sddot[1], 0.0])
    F_react_body -= m_d * sddot_3d

    # Convert to inertial frame
    F_reaction_inertial = R_L @ F_react_body

    # Torque about load CM in body frame
    tau_reaction_body = np.cross(rho, F_react_body)

    return F_reaction_inertial, tau_reaction_body
