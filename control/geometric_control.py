"""
control/geometric_control.py — SE(3) geometric tracking controller.

Implements the geometric controller from:
    Lee, Leok & McClamroch, "Geometric Tracking Control of a Quadrotor UAV
    on SE(3)", CDC 2010.

This controller works directly on the Lie group SO(3) (via rotation matrices)
— no Euler angles, no gimbal lock, no singularities.  It produces:
  • A desired thrust magnitude (scalar) and desired attitude (rotation matrix).
  • A body-frame torque command from the attitude error.

The same controller structure is used for both individual drones and for the
load-level controller in centralized mode.
"""

from __future__ import annotations

import numpy as np
from dynamics.drone import hat, vee, quat_to_rotmat


def compute_desired_thrust_and_attitude(
    p: np.ndarray,
    v: np.ndarray,
    p_des: np.ndarray,
    v_des: np.ndarray,
    a_des: np.ndarray,
    yaw_des: float,
    mass: float,
    gravity: float,
    Kp: np.ndarray,
    Kd: np.ndarray,
    F_feedforward: np.ndarray = None,
) -> tuple[float, np.ndarray]:
    """Position control → desired thrust magnitude + desired rotation matrix.

    Parameters
    ----------
    p, v : current position and velocity (inertial frame)
    p_des, v_des, a_des : desired position, velocity, acceleration
    yaw_des : desired yaw angle (rad)
    mass : body mass (kg)
    gravity : gravitational acceleration
    Kp, Kd : (3,) position and velocity gain vectors
    F_feedforward : (3,) optional feedforward force (e.g. expected cable tension)

    Returns
    -------
    F_thrust : scalar thrust magnitude
    R_des : (3,3) desired rotation matrix
    """
    e3 = np.array([0.0, 0.0, 1.0])

    # Position error
    e_p = p - p_des
    e_v = v - v_des

    # Desired force vector (inertial frame)
    F_des = (-np.diag(Kp) @ e_p - np.diag(Kd) @ e_v
             + mass * gravity * e3 + mass * a_des)

    if F_feedforward is not None:
        F_des += F_feedforward

    # Desired thrust magnitude (projection onto body z-axis direction)
    F_thrust = np.linalg.norm(F_des)
    if F_thrust < 1e-6:
        F_thrust = mass * gravity
        F_des = mass * gravity * e3

    # Desired body z-axis
    b3_des = F_des / np.linalg.norm(F_des)

    # Desired body x-axis (from yaw)
    b1_yaw = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

    # Gram-Schmidt to construct orthonormal frame
    b2_des = np.cross(b3_des, b1_yaw)
    b2_norm = np.linalg.norm(b2_des)
    if b2_norm < 1e-6:
        # b3_des is parallel to b1_yaw — use fallback
        b1_yaw = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0])
        b2_des = np.cross(b3_des, b1_yaw)
        b2_norm = np.linalg.norm(b2_des)

    b2_des = b2_des / b2_norm
    b1_des = np.cross(b2_des, b3_des)

    R_des = np.column_stack([b1_des, b2_des, b3_des])

    return F_thrust, R_des


def attitude_torque_command(
    R: np.ndarray,
    omega: np.ndarray,
    R_des: np.ndarray,
    omega_des: np.ndarray,
    omega_dot_des: np.ndarray,
    J: np.ndarray,
    k_R: float,
    k_omega: float,
) -> np.ndarray:
    """Geometric SO(3) attitude controller.

    Computes body-frame torque command:
        τ = −k_R e_R − k_Ω e_Ω + ω × Jω
            − J(ω̂ Rᵀ R_des ω_des − Rᵀ R_des ω̇_des)

    Parameters
    ----------
    R : (3,3) current rotation matrix (body → inertial)
    omega : (3,) current angular velocity (body frame)
    R_des : (3,3) desired rotation matrix
    omega_des : (3,) desired angular velocity (body frame of desired)
    omega_dot_des : (3,) desired angular acceleration
    J : (3,3) inertia tensor
    k_R : attitude proportional gain
    k_omega : attitude derivative gain

    Returns
    -------
    tau_cmd : (3,) body-frame torque command
    """
    # Attitude error on SO(3)
    # e_R = 0.5 * vee(R_des^T R - R^T R_des)
    e_R_matrix = R_des.T @ R - R.T @ R_des
    e_R = 0.5 * vee(e_R_matrix)

    # Angular velocity error
    e_omega = omega - R.T @ R_des @ omega_des

    # Feedforward term
    ff = np.cross(omega, J @ omega)
    ff -= J @ (hat(omega) @ R.T @ R_des @ omega_des - R.T @ R_des @ omega_dot_des)

    # Torque command
    tau_cmd = -k_R * e_R - k_omega * e_omega + ff

    return tau_cmd


def compute_load_desired_wrench(
    p_L: np.ndarray,
    v_L: np.ndarray,
    R_L: np.ndarray,
    omega_L: np.ndarray,
    p_L_des: np.ndarray,
    v_L_des: np.ndarray,
    a_L_des: np.ndarray,
    R_L_des: np.ndarray,
    omega_L_des: np.ndarray,
    omega_L_dot_des: np.ndarray,
    total_mass: float,
    J_L: np.ndarray,
    gravity: float,
    Kp: np.ndarray,
    Kd: np.ndarray,
    k_R: float,
    k_omega: float,
) -> np.ndarray:
    """Compute desired wrench [F_des; τ_des] ∈ R⁶ on the load.

    Used by Mode B (centralized allocation).

    Returns
    -------
    wrench_des : (6,) array [F_x, F_y, F_z, τ_x, τ_y, τ_z] in inertial frame
                 (forces) and body frame (torques).
    """
    e3 = np.array([0.0, 0.0, 1.0])

    # Translational
    e_p = p_L - p_L_des
    e_v = v_L - v_L_des
    F_des = (-np.diag(Kp) @ e_p - np.diag(Kd) @ e_v
             + total_mass * gravity * e3 + total_mass * a_L_des)

    # Rotational (in body frame)
    tau_des = attitude_torque_command(
        R_L, omega_L, R_L_des, omega_L_des, omega_L_dot_des,
        J_L, k_R, k_omega,
    )

    return np.concatenate([F_des, tau_des])
