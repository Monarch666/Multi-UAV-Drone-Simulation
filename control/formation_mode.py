"""
control/formation_mode.py — Decentralized (Mode A) and centralized (Mode B) controllers.

Mode A — Decentralized Formation Tracking
    Each drone independently tracks a virtual point above its assigned basket
    corner.  The virtual point follows the load's desired trajectory, offset
    by the nominal cable geometry.  Each drone runs a cascaded SE(3) controller
    with cable-tension feedforward.

Mode B — Centralized Load-Wrench Allocation
    A single load-level SE(3) controller computes the desired wrench on the
    load.  The QP allocator distributes this wrench to 4 cable tensions.
    Each drone's local controller then tracks the position/attitude that
    produces its allocated tension.
"""

from __future__ import annotations

import numpy as np
from dynamics.drone import (
    DroneParams, unpack_drone_state, quat_to_rotmat,
    get_cable_attach_world, get_cable_attach_velocity_world,
)
from dynamics.load import (
    BasketParams, unpack_load_state, get_corner_world,
)
from dynamics.sliding_mass import unpack_sliding_state
from dynamics.system import SystemConfig, NUM_DRONES
from control.geometric_control import (
    compute_desired_thrust_and_attitude,
    attitude_torque_command,
    compute_load_desired_wrench,
)
from control.allocation_qp import (
    build_structure_matrix, allocate_tensions, compute_static_load_sharing,
)
from control.mixer import wrench_to_rotor_speeds


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def trajectory_hover(t: float, cfg: dict) -> tuple:
    """Hover at a fixed position."""
    pos = np.array(cfg.get('hover_pos', [0.0, 0.0, 3.0]), dtype=float)
    vel = np.zeros(3)
    acc = np.zeros(3)
    yaw = 0.0
    return pos, vel, acc, yaw


def trajectory_circle(t: float, cfg: dict) -> tuple:
    """Circular trajectory in the horizontal plane."""
    center = np.array(cfg.get('circle_center', [0.0, 0.0, 3.0]), dtype=float)
    r = cfg.get('circle_radius', 1.5)
    w = cfg.get('circle_omega', 0.3)
    yaw_rate = cfg.get('circle_yaw_rate', 0.0)

    pos = center + np.array([r * np.cos(w*t), r * np.sin(w*t), 0.0])
    vel = np.array([-r*w * np.sin(w*t), r*w * np.cos(w*t), 0.0])
    acc = np.array([-r*w**2 * np.cos(w*t), -r*w**2 * np.sin(w*t), 0.0])
    yaw = yaw_rate * t

    return pos, vel, acc, yaw


def trajectory_lemniscate(t: float, cfg: dict) -> tuple:
    """Figure-8 (lemniscate of Bernoulli) trajectory."""
    center = np.array(cfg.get('circle_center', [0.0, 0.0, 3.0]), dtype=float)
    r = cfg.get('circle_radius', 1.5)
    w = cfg.get('circle_omega', 0.3)

    pos = center + np.array([
        r * np.sin(w*t),
        r * np.sin(w*t) * np.cos(w*t),
        0.0
    ])
    vel = np.array([
        r*w * np.cos(w*t),
        r*w * (np.cos(w*t)**2 - np.sin(w*t)**2),
        0.0
    ])
    acc = np.array([
        -r*w**2 * np.sin(w*t),
        -4*r*w**2 * np.sin(w*t) * np.cos(w*t),
        0.0
    ])
    yaw = 0.0
    return pos, vel, acc, yaw


def trajectory_step(t: float, cfg: dict) -> tuple:
    """Step input: hover, then move to a new position."""
    start = np.array(cfg.get('hover_pos', [0.0, 0.0, 3.0]), dtype=float)
    step_time = 5.0
    step_offset = np.array([1.0, 0.0, 0.5])

    if t < step_time:
        return start, np.zeros(3), np.zeros(3), 0.0
    else:
        return start + step_offset, np.zeros(3), np.zeros(3), 0.0


TRAJECTORY_MAP = {
    'hover': trajectory_hover,
    'circle': trajectory_circle,
    'lemniscate': trajectory_lemniscate,
    'step': trajectory_step,
}


# ---------------------------------------------------------------------------
# Control parameters (extracted from config)
# ---------------------------------------------------------------------------

class ControlParams:
    """Extracted control parameters."""
    def __init__(self, ctrl_cfg: dict, traj_cfg: dict):
        self.Kp = np.array(ctrl_cfg.get('Kp_pos', [8.0, 8.0, 12.0]))
        self.Kd = np.array(ctrl_cfg.get('Kd_pos', [5.0, 5.0, 8.0]))
        self.k_R = ctrl_cfg.get('k_R', 15.0)
        self.k_omega = ctrl_cfg.get('k_omega', 3.5)
        self.Kp_load = np.array(ctrl_cfg.get('Kp_load', [6.0, 6.0, 10.0]))
        self.Kd_load = np.array(ctrl_cfg.get('Kd_load', [4.0, 4.0, 6.0]))
        self.k_R_load = ctrl_cfg.get('k_R_load', 12.0)
        self.k_omega_load = ctrl_cfg.get('k_omega_load', 3.0)
        self.T_min = ctrl_cfg.get('T_min', 0.5)
        self.T_max = ctrl_cfg.get('T_max', 40.0)
        self.desired_yaw = ctrl_cfg.get('desired_yaw', 0.0)
        self.mode = ctrl_cfg.get('mode', 'formation')

        self.traj_type = traj_cfg.get('type', 'hover')
        self.traj_cfg = traj_cfg


# ---------------------------------------------------------------------------
# Mode A: Decentralized formation tracking
# ---------------------------------------------------------------------------

def controller_formation(
    t: float,
    drone_states: list[np.ndarray],
    load_state: np.ndarray,
    sliding_state: np.ndarray,
    cfg: SystemConfig,
    ctrl: ControlParams,
) -> list[np.ndarray]:
    """Mode A controller: each drone tracks its corner virtual point.

    Returns list of 4 × (4,) rotor speed commands.
    """
    # Get desired load trajectory
    traj_fn = TRAJECTORY_MAP.get(ctrl.traj_type, trajectory_hover)
    p_L_des, v_L_des, a_L_des, yaw_des = traj_fn(t, ctrl.traj_cfg)

    p_L, v_L, q_L, omega_L = unpack_load_state(load_state)
    q_L = q_L / np.linalg.norm(q_L)
    R_L = quat_to_rotmat(q_L)
    R_L_des = np.eye(3)  # desired load attitude = level

    bp = cfg.basket_params
    rotor_cmds = []

    # Cache static tensions to avoid solving load allocation at every single ODE derivative evaluation
    if not hasattr(ctrl, '_T_static_cached'):
        total_weight = (bp.total_mass + bp.dynamic_load_mass) * cfg.gravity
        # Assume nominal cable directions are straight up at hover
        n_hats_nominal = [np.array([0.0, 0.0, 1.0]) for _ in range(NUM_DRONES)]
        ctrl._T_static_cached = compute_static_load_sharing(
            bp.corner_points_body, n_hats_nominal, total_weight, np.eye(3)
        )
    T_ff = ctrl._T_static_cached

    n_hats = []
    for i in range(NUM_DRONES):
        a_i = get_cable_attach_world(drone_states[i], cfg.drone_params[i])
        r_i = get_corner_world(p_L, R_L, bp.corner_points_body[i])
        e = a_i - r_i
        L = np.linalg.norm(e)
        n_hats.append(e / max(L, 1e-6))

    for i in range(NUM_DRONES):
        dp = cfg.drone_params[i]
        ds = drone_states[i]
        p_i, v_i, q_i, omega_i, rotors_i = unpack_drone_state(ds)
        q_i = q_i / np.linalg.norm(q_i)
        R_i = quat_to_rotmat(q_i)

        # Desired position: above corner, at cable length
        corner_body = bp.corner_points_body[i]
        L0 = cfg.cable_params[i].natural_length
        # Desired corner position in world (from desired load trajectory)
        corner_des_world = p_L_des + R_L_des @ corner_body
        p_i_des = corner_des_world + np.array([0.0, 0.0, L0])
        # Account for cable attach offset
        p_i_des -= dp.cable_attach_offset  # body ≈ inertial at hover

        v_i_des = v_L_des.copy()
        a_i_des = a_L_des.copy()

        # Cable tension feedforward (expected pull direction × expected tension)
        F_ff = T_ff[i] * n_hats[i]  # positive upward force to balance cable pull

        # Position controller
        F_thrust, R_des = compute_desired_thrust_and_attitude(
            p_i, v_i, p_i_des, v_i_des, a_i_des,
            ctrl.desired_yaw, dp.mass, cfg.gravity,
            ctrl.Kp, ctrl.Kd,
            F_feedforward=F_ff,
        )

        # Attitude controller
        tau_cmd = attitude_torque_command(
            R_i, omega_i, R_des, np.zeros(3), np.zeros(3),
            dp.J, ctrl.k_R, ctrl.k_omega,
        )

        # Mix to rotor speeds
        omega_cmd = wrench_to_rotor_speeds(
            F_thrust, tau_cmd, dp.arm_length,
            dp.k_T, dp.k_Q, dp.omega_min, dp.omega_max,
        )

        rotor_cmds.append(omega_cmd)

    return rotor_cmds


# ---------------------------------------------------------------------------
# Mode B: Centralized load-wrench allocation
# ---------------------------------------------------------------------------

def controller_centralized(
    t: float,
    drone_states: list[np.ndarray],
    load_state: np.ndarray,
    sliding_state: np.ndarray,
    cfg: SystemConfig,
    ctrl: ControlParams,
) -> list[np.ndarray]:
    """Mode B controller: centralized wrench allocation.

    Returns list of 4 × (4,) rotor speed commands.
    """
    # Get desired load trajectory
    traj_fn = TRAJECTORY_MAP.get(ctrl.traj_type, trajectory_hover)
    p_L_des, v_L_des, a_L_des, yaw_des = traj_fn(t, ctrl.traj_cfg)

    p_L, v_L, q_L, omega_L = unpack_load_state(load_state)
    q_L = q_L / np.linalg.norm(q_L)
    R_L = quat_to_rotmat(q_L)

    bp = cfg.basket_params

    # Desired load attitude (level)
    R_L_des = np.eye(3)

    # Compute desired wrench on load
    total_mass_with_dynamic = bp.total_mass + bp.dynamic_load_mass
    wrench_des = compute_load_desired_wrench(
        p_L, v_L, R_L, omega_L,
        p_L_des, v_L_des, a_L_des,
        R_L_des, np.zeros(3), np.zeros(3),
        total_mass_with_dynamic,
        bp.J_combined, cfg.gravity,
        ctrl.Kp_load, ctrl.Kd_load,
        ctrl.k_R_load, ctrl.k_omega_load,
    )

    # Build structure matrix from current cable geometry
    n_hats = []
    for i in range(NUM_DRONES):
        a_i = get_cable_attach_world(drone_states[i], cfg.drone_params[i])
        r_i = get_corner_world(p_L, R_L, bp.corner_points_body[i])
        e = a_i - r_i
        L = np.linalg.norm(e)
        n_hats.append(e / max(L, 1e-6))

    A = build_structure_matrix(
        n_hats, list(bp.corner_points_body), R_L
    )

    # Allocate tensions
    T_alloc, feasible = allocate_tensions(
        A, wrench_des, ctrl.T_min, ctrl.T_max
    )

    # Each drone tracks position to produce its allocated tension
    rotor_cmds = []
    for i in range(NUM_DRONES):
        dp = cfg.drone_params[i]
        ds = drone_states[i]
        p_i, v_i, q_i, omega_i, rotors_i = unpack_drone_state(ds)
        q_i = q_i / np.linalg.norm(q_i)
        R_i = quat_to_rotmat(q_i)

        # Desired drone position: corner + cable length above
        corner_body = bp.corner_points_body[i]
        L0 = cfg.cable_params[i].natural_length
        corner_des_world = p_L_des + R_L_des @ corner_body
        p_i_des = corner_des_world + np.array([0.0, 0.0, L0])
        p_i_des -= dp.cable_attach_offset

        v_i_des = v_L_des.copy()
        a_i_des = a_L_des.copy()

        # Feedforward: allocated cable tension
        F_ff = T_alloc[i] * n_hats[i]

        # Position controller
        F_thrust, R_des = compute_desired_thrust_and_attitude(
            p_i, v_i, p_i_des, v_i_des, a_i_des,
            ctrl.desired_yaw, dp.mass, cfg.gravity,
            ctrl.Kp, ctrl.Kd,
            F_feedforward=F_ff,
        )

        # Attitude controller
        tau_cmd = attitude_torque_command(
            R_i, omega_i, R_des, np.zeros(3), np.zeros(3),
            dp.J, ctrl.k_R, ctrl.k_omega,
        )

        # Mix to rotor speeds
        omega_cmd = wrench_to_rotor_speeds(
            F_thrust, tau_cmd, dp.arm_length,
            dp.k_T, dp.k_Q, dp.omega_min, dp.omega_max,
        )

        rotor_cmds.append(omega_cmd)

    return rotor_cmds


# ---------------------------------------------------------------------------
# Controller factory — returns the controller function for system.py
# ---------------------------------------------------------------------------

def make_controller(ctrl_cfg: dict, traj_cfg: dict):
    """Create a controller callable for use with make_system_rhs.

    Returns a function with signature:
        controller(t, drone_states, load_state, sliding_state, cfg) → rotor_cmds
    """
    ctrl = ControlParams(ctrl_cfg, traj_cfg)

    if ctrl.mode == "centralized":
        def controller(t, drone_states, load_state, sliding_state, cfg):
            return controller_centralized(
                t, drone_states, load_state, sliding_state, cfg, ctrl
            )
    else:  # "formation" (default)
        def controller(t, drone_states, load_state, sliding_state, cfg):
            return controller_formation(
                t, drone_states, load_state, sliding_state, cfg, ctrl
            )

    return controller
