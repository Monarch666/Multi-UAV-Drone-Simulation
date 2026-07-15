"""
dynamics/drone.py — Single quadrotor UAV dynamics (Newton-Euler formulation).

Coordinate convention
---------------------
Inertial frame: ENU (x-East, y-North, z-Up), right-handed.
Body frame per UAV: x-forward, y-left, z-up (right-handed, consistent with ENU).
Rotation matrix R ∈ SO(3) maps body→inertial:  v_inertial = R @ v_body.
Attitude is stored as a unit quaternion q = [qw, qx, qy, qz] for integration
(avoids gimbal-lock and maintains orthogonality cheaply via renormalisation).

State per drone (24 elements):
    p      (3)  — position in inertial frame
    v      (3)  — velocity in inertial frame
    q      (4)  — unit quaternion (scalar-first: [w, x, y, z])
    omega  (3)  — angular velocity in BODY frame
    rotors (4)  — current rotor speeds (rad/s), with first-order motor lag

References
----------
Lee, Leok & McClamroch, "Geometric Tracking Control of a Quadrotor UAV on
SE(3)", CDC 2010.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers — quaternion / rotation utilities
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to unit quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.
    """
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Time derivative of quaternion given body angular velocity.

    dq/dt = 0.5 * q ⊗ [0, ω_body]
    """
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_multiply(q, omega_quat)


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (hat) matrix of a 3-vector."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


def vee(S: np.ndarray) -> np.ndarray:
    """Inverse of the hat map: extract 3-vector from a skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# ---------------------------------------------------------------------------
# Drone parameters and state
# ---------------------------------------------------------------------------

@dataclass
class DroneParams:
    """Physical parameters for a single quadrotor."""
    mass: float = 2.0                       # kg
    J: np.ndarray = field(default_factory=lambda: np.diag([0.02, 0.02, 0.04]))
    J_inv: np.ndarray = field(default=None)
    arm_length: float = 0.25               # m
    k_T: float = 1.5e-5                    # N / (rad/s)^2
    k_Q: float = 2.5e-7                    # N·m / (rad/s)^2
    tau_motor: float = 0.03                # motor time constant (s)
    omega_max: float = 900.0               # max rotor speed (rad/s)
    omega_min: float = 50.0                # min (idle) rotor speed (rad/s)
    cable_attach_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.05])
    )                                       # body-frame offset of cable attach point
    drag_coeff: float = 0.15               # quadratic drag coefficient

    def __post_init__(self):
        self.J = np.asarray(self.J, dtype=float)
        if self.J.ndim == 1:
            self.J = np.diag(self.J)
        self.J_inv = np.linalg.inv(self.J)
        self.cable_attach_offset = np.asarray(self.cable_attach_offset, dtype=float)


# ---------------------------------------------------------------------------
# Mixer — maps rotor speeds² to thrust + body torques and back
# ---------------------------------------------------------------------------

def build_mixer_matrix(arm_length: float, k_T: float, k_Q: float) -> np.ndarray:
    """Build the 4×4 mixer matrix for an X-configuration quadrotor.

    Rotor layout (top-view, body frame x-forward, y-left):
        Rotor 1: front-right (+x, -y)  — CW   (produces +yaw torque via reaction)
        Rotor 2: front-left  (+x, +y)  — CCW  (produces -yaw torque)
        Rotor 3: rear-left   (-x, +y)  — CW
        Rotor 4: rear-right  (-x, -y)  — CCW

    Returns M such that [F_total, τ_x, τ_y, τ_z]ᵀ = M @ [ω₁², ω₂², ω₃², ω₄²]ᵀ
    """
    L = arm_length / np.sqrt(2)   # effective moment arm for X-frame
    return np.array([
        [ k_T,     k_T,     k_T,     k_T    ],   # total thrust
        [-k_T*L,   k_T*L,   k_T*L,  -k_T*L  ],   # roll torque (τ_x)
        [ k_T*L,   k_T*L,  -k_T*L,  -k_T*L  ],   # pitch torque (τ_y)
        [ k_Q,    -k_Q,     k_Q,    -k_Q     ],   # yaw torque (τ_z)
    ])


# ---------------------------------------------------------------------------
# State packing / unpacking
# ---------------------------------------------------------------------------

DRONE_STATE_DIM = 17  # 3 + 3 + 4 + 3 + 4


def pack_drone_state(p: np.ndarray, v: np.ndarray, q: np.ndarray,
                     omega: np.ndarray, rotors: np.ndarray) -> np.ndarray:
    """Pack drone state into a flat array (17 elements)."""
    return np.concatenate([p, v, q, omega, rotors])


def unpack_drone_state(x: np.ndarray):
    """Unpack flat array into (p, v, q, omega, rotors)."""
    p     = x[0:3]
    v     = x[3:6]
    q     = x[6:10]
    omega = x[10:13]
    rotors = x[13:17]
    return p, v, q, omega, rotors


# ---------------------------------------------------------------------------
# Single-drone equations of motion
# ---------------------------------------------------------------------------

def drone_derivatives(
    state: np.ndarray,
    params: DroneParams,
    rotor_speed_cmds: np.ndarray,
    F_ext: np.ndarray,         # external force in INERTIAL frame (e.g. cable)
    tau_ext_body: np.ndarray,  # external torque in BODY frame (e.g. cable torque)
    gravity: float = 9.80665,
    wind_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute state derivative for a single quadrotor.

    Parameters
    ----------
    state : (17,) array — packed drone state
    params : DroneParams
    rotor_speed_cmds : (4,) array — commanded rotor speeds (rad/s)
    F_ext : (3,) — external force in inertial frame
    tau_ext_body : (3,) — external torque in body frame
    gravity : scalar
    wind_velocity : (3,) optional — wind velocity in inertial frame

    Returns
    -------
    dstate : (17,) array — time derivatives
    """
    p, v, q, omega, rotors = unpack_drone_state(state)

    # Renormalise quaternion to stay on SO(3) manifold
    q = q / np.linalg.norm(q)
    R = quat_to_rotmat(q)

    # ---- Motor dynamics (first-order lag) ----
    # Clamp commands to [omega_min, omega_max]
    cmds_clamped = np.clip(rotor_speed_cmds, params.omega_min, params.omega_max)
    d_rotors = (cmds_clamped - rotors) / params.tau_motor

    # ---- Thrust and torques from rotors ----
    omega_sq = rotors ** 2
    mixer = build_mixer_matrix(params.arm_length, params.k_T, params.k_Q)
    wrench = mixer @ omega_sq   # [F_total, τ_x, τ_y, τ_z]
    F_thrust_total = wrench[0]
    tau_rotors = wrench[1:4]    # body-frame torque

    # Thrust acts along body z-axis (upward in body frame)
    thrust_body = np.array([0.0, 0.0, F_thrust_total])
    thrust_inertial = R @ thrust_body

    # ---- Aerodynamic drag ----
    v_rel = v.copy()
    if wind_velocity is not None:
        v_rel = v - wind_velocity
    speed = np.linalg.norm(v_rel)
    F_drag = -params.drag_coeff * speed * v_rel  # quadratic drag

    # ---- Translational dynamics (Newton) ----
    # F = m*a  →  a = (1/m) * (gravity + thrust + cable + drag)
    gravity_force = np.array([0.0, 0.0, -params.mass * gravity])
    dp = v
    dv = (gravity_force + thrust_inertial + F_ext + F_drag) / params.mass

    # ---- Rotational dynamics (Euler) ----
    # J ω̇ = -ω × (Jω) + τ_rotors + τ_ext
    Jomega = params.J @ omega
    domega = params.J_inv @ (
        -np.cross(omega, Jomega) + tau_rotors + tau_ext_body
    )

    # ---- Quaternion kinematics ----
    dq = quat_derivative(q, omega)

    return np.concatenate([dp, dv, dq, domega, d_rotors])


def get_cable_attach_world(state: np.ndarray, params: DroneParams) -> np.ndarray:
    """World-frame position of the cable attachment point on the drone.

    a_i = p_i + R_i · d_i
    """
    p, _, q, _, _ = unpack_drone_state(state)
    q = q / np.linalg.norm(q)
    R = quat_to_rotmat(q)
    return p + R @ params.cable_attach_offset


def get_cable_attach_velocity_world(
    state: np.ndarray, params: DroneParams
) -> np.ndarray:
    """World-frame velocity of the cable attachment point.

    ȧ_i = v_i + ω_i × (R_i · d_i)   (expressed in inertial frame)
    """
    _, v, q, omega, _ = unpack_drone_state(state)
    q = q / np.linalg.norm(q)
    R = quat_to_rotmat(q)
    r_offset_world = R @ params.cable_attach_offset
    # ω is in body frame; angular velocity in world frame: Ω = R @ ω
    omega_world = R @ omega
    return v + np.cross(omega_world, r_offset_world)


def cable_torque_on_drone_body(
    state: np.ndarray, params: DroneParams, F_cable_inertial: np.ndarray
) -> np.ndarray:
    """Compute cable torque on drone in BODY frame.

    τ_cable = d_i × (Rᵀ F_cable)  in body frame
    equivalently:  Rᵀ [(R d_i) × F_cable]  but we keep it simple.
    """
    _, _, q, _, _ = unpack_drone_state(state)
    q = q / np.linalg.norm(q)
    R = quat_to_rotmat(q)
    F_body = R.T @ F_cable_inertial
    return np.cross(params.cable_attach_offset, F_body)
