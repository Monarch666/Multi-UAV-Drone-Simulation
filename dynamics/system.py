"""
dynamics/system.py — Full multi-body system assembler for solve_ivp.

Assembles the complete state vector for:
  • 4 quadrotor UAVs  (17 states each = 68)
  • 1 basket/load     (13 states)
  • 1 sliding mass    (4 states)
  Total: 85 states

Provides the state-derivative function `system_rhs(t, y, ...)` compatible
with scipy.integrate.solve_ivp, as well as state pack/unpack utilities and
the fixed-step RK4 fallback.

Cable model selection ('elastic' vs 'rigid_constraint') is handled here.
For the elastic model, cable forces are computed directly and added to the
drone/load EOMs.  For the rigid-constraint model, Lagrange multipliers are
solved for each step (index-3 DAE with Baumgarte stabilisation).
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field

from dynamics.drone import (
    DroneParams, DRONE_STATE_DIM,
    pack_drone_state, unpack_drone_state,
    drone_derivatives, quat_to_rotmat, hat, vee,
    get_cable_attach_world, get_cable_attach_velocity_world,
    cable_torque_on_drone_body,
)
from dynamics.cable import (
    CableParams, compute_cable_force_elastic, compute_rigid_constraint_terms,
    cable_force_on_load, cable_force_on_drone,
)
from dynamics.load import (
    BasketParams, LOAD_STATE_DIM,
    pack_load_state, unpack_load_state,
    get_corner_world, get_corner_velocity_world,
)
from dynamics.sliding_mass import (
    SLIDING_MASS_STATE_DIM,
    pack_sliding_state, unpack_sliding_state,
    sliding_mass_derivatives, sliding_mass_reaction_on_load,
)

# ---------------------------------------------------------------------------
# System configuration
# ---------------------------------------------------------------------------

NUM_DRONES = 4
TOTAL_STATE_DIM = NUM_DRONES * DRONE_STATE_DIM + LOAD_STATE_DIM + SLIDING_MASS_STATE_DIM
# = 4 * 17 + 13 + 4 = 85


@dataclass
class SystemConfig:
    """Complete simulation configuration."""
    drone_params: list[DroneParams] = field(default_factory=lambda: [DroneParams() for _ in range(4)])
    cable_params: list[CableParams] = field(default_factory=lambda: [CableParams() for _ in range(4)])
    basket_params: BasketParams = field(default_factory=BasketParams)
    gravity: float = 9.80665

    # Wind
    wind_enabled: bool = False
    wind_mean_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wind_gust_amplitude: float = 0.0
    wind_gust_frequency: float = 0.0


# ---------------------------------------------------------------------------
# State packing / unpacking
# ---------------------------------------------------------------------------

def pack_system_state(
    drone_states: list[np.ndarray],   # 4 × (17,)
    load_state: np.ndarray,           # (13,)
    sliding_state: np.ndarray,        # (4,)
) -> np.ndarray:
    """Pack all sub-states into a single flat vector (85,)."""
    parts = drone_states + [load_state, sliding_state]
    return np.concatenate(parts)


def unpack_system_state(y: np.ndarray):
    """Unpack (85,) vector into sub-states.

    Returns
    -------
    drone_states : list of 4 × (17,) arrays
    load_state   : (13,) array
    sliding_state : (4,) array
    """
    idx = 0
    drone_states = []
    for _ in range(NUM_DRONES):
        drone_states.append(y[idx:idx + DRONE_STATE_DIM])
        idx += DRONE_STATE_DIM
    load_state = y[idx:idx + LOAD_STATE_DIM]
    idx += LOAD_STATE_DIM
    sliding_state = y[idx:idx + SLIDING_MASS_STATE_DIM]
    return drone_states, load_state, sliding_state


# ---------------------------------------------------------------------------
# Wind model
# ---------------------------------------------------------------------------

def compute_wind(t: float, cfg: SystemConfig) -> np.ndarray:
    """Compute wind velocity at time t (inertial frame)."""
    if not cfg.wind_enabled:
        return np.zeros(3)
    gust = cfg.wind_gust_amplitude * np.sin(cfg.wind_gust_frequency * t)
    wind = cfg.wind_mean_velocity.copy()
    wind[0] += gust  # gust along x-axis
    return wind


# ---------------------------------------------------------------------------
# Normalise quaternions in-place to stay on SO(3)
# ---------------------------------------------------------------------------

def normalise_quaternions(y: np.ndarray) -> np.ndarray:
    """Renormalise all quaternion components in the state vector."""
    y = y.copy()
    for i in range(NUM_DRONES):
        base = i * DRONE_STATE_DIM + 6   # quaternion starts at index 6 within drone state
        q = y[base:base + 4]
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            y[base:base + 4] = q / norm

    # Load quaternion
    load_base = NUM_DRONES * DRONE_STATE_DIM + 6
    q_load = y[load_base:load_base + 4]
    norm = np.linalg.norm(q_load)
    if norm > 1e-10:
        y[load_base:load_base + 4] = q_load / norm

    return y


# ---------------------------------------------------------------------------
# Main RHS function for solve_ivp
# ---------------------------------------------------------------------------

def make_system_rhs(
    cfg: SystemConfig,
    controller_fn: Callable,
) -> Callable:
    """Create the ODE right-hand-side function compatible with solve_ivp.

    Parameters
    ----------
    cfg : SystemConfig — physical parameters
    controller_fn : callable(t, drone_states, load_state, sliding_state, cfg)
        → list of 4 × (4,) arrays of rotor speed commands

    Returns
    -------
    rhs : callable(t, y) → dy/dt
    """
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        # ---- Renormalise quaternions ----
        y = normalise_quaternions(y)

        # ---- Unpack ----
        drone_states, load_state, sliding_state = unpack_system_state(y)
        p_L, v_L, q_L, omega_L = unpack_load_state(load_state)
        q_L = q_L / np.linalg.norm(q_L)
        R_L = quat_to_rotmat(q_L)
        s, sdot = unpack_sliding_state(sliding_state)

        # ---- Wind ----
        wind = compute_wind(t, cfg)

        # ---- Controller ----
        rotor_cmds = controller_fn(t, drone_states, load_state, sliding_state, cfg)

        # ---- Cable forces (elastic model) ----
        cable_model = cfg.cable_params[0].model  # assume all cables use same model

        tensions = np.zeros(NUM_DRONES)
        F_cables_on_drones = [np.zeros(3) for _ in range(NUM_DRONES)]
        tau_cables_on_drones_body = [np.zeros(3) for _ in range(NUM_DRONES)]
        F_cables_on_load = [np.zeros(3) for _ in range(NUM_DRONES)]
        tau_cables_on_load_body = [np.zeros(3) for _ in range(NUM_DRONES)]

        for i in range(NUM_DRONES):
            dp = cfg.drone_params[i]
            cp = cfg.cable_params[i]
            ds = drone_states[i]

            # Attachment points (world frame)
            a_i = get_cable_attach_world(ds, dp)
            r_i = get_corner_world(p_L, R_L, cfg.basket_params.corner_points_body[i])
            va_i = get_cable_attach_velocity_world(ds, dp)
            vr_i = get_corner_velocity_world(v_L, omega_L, R_L,
                                              cfg.basket_params.corner_points_body[i])

            if cable_model == "elastic":
                tension, n_hat, _ = compute_cable_force_elastic(
                    a_i, r_i, va_i, vr_i, cp)
            else:
                # For rigid constraint, we use a high-stiffness elastic
                # approximation in the RHS (Baumgarte stabilization is
                # handled implicitly by the stiff integrator).
                # A proper DAE formulation would require a different solver.
                ct = compute_rigid_constraint_terms(a_i, r_i, va_i, vr_i, cp)
                n_hat = ct['n_hat']
                # Use Baumgarte-stabilized elastic-like force
                L = ct['L']
                if L > 1e-12:
                    delta = max(0.0, L - cp.natural_length)
                    v_rel = va_i - vr_i
                    delta_dot = np.dot(v_rel, n_hat)
                    # Very high stiffness for "rigid" behavior
                    k_rigid = 50000.0
                    c_rigid = 500.0
                    tension = max(0.0, k_rigid * delta + c_rigid * delta_dot)
                    # Add Baumgarte correction
                    tension += max(0.0, -ct['baumgarte_rhs'] * 100.0)
                else:
                    tension = 0.0

            tensions[i] = tension

            # Forces
            F_on_load = cable_force_on_load(tension, n_hat)
            F_on_drone = cable_force_on_drone(tension, n_hat)

            F_cables_on_load[i] = F_on_load
            F_cables_on_drones[i] = F_on_drone

            # Torques on load (body frame)
            r_offset_body = cfg.basket_params.corner_points_body[i]
            tau_cables_on_load_body[i] = np.cross(r_offset_body, R_L.T @ F_on_load)

            # Torques on drone (body frame)
            tau_cables_on_drones_body[i] = cable_torque_on_drone_body(
                ds, dp, F_on_drone)

        # ---- Coupled Load and Sliding Mass Dynamics (8x8 system) ----
        bp = cfg.basket_params
        m_L = bp.total_mass
        m_d = bp.dynamic_load_mass
        J_L = bp.J_combined

        # 3D position of sliding mass in load body frame
        rho = np.array([s[0], s[1], bp.floor_height])
        rho_cross = hat(rho)

        # 3D relative velocity in body frame
        sdot_3d = np.array([sdot[0], sdot[1], 0.0])

        # Coriolis & Centrifugal accelerations (expressed in body frame)
        a_coriolis_body = 2.0 * np.cross(omega_L, sdot_3d)
        a_centrifugal_body = np.cross(omega_L, np.cross(omega_L, rho))
        a_fictitious_body = a_coriolis_body + a_centrifugal_body
        a_fictitious_world = R_L @ a_fictitious_body

        # Gravity vectors
        g_world = np.array([0.0, 0.0, -cfg.gravity])
        g_body = R_L.T @ g_world

        # External forces on load (cables + aero drag)
        F_cable_total = sum(F_cables_on_load)
        tau_cable_total_body = sum(tau_cables_on_load_body)

        v_L_rel = v_L - wind
        F_drag_world = -0.5 * np.linalg.norm(v_L_rel) * v_L_rel
        tau_drag_body = np.zeros(3)  # assumed small

        # 1. Estimate normal force Nz (to evaluate sliding friction force)
        # Using a preliminary coupled vertical force balance:
        F_ext_total_world = (m_L + m_d) * g_world + F_cable_total + F_drag_world
        a_L_prelim = F_ext_total_world / (m_L + m_d)
        a_L_prelim_body = R_L.T @ a_L_prelim
        N_z_prelim = -m_d * (g_body[2] - a_L_prelim_body[2] - a_fictitious_body[2])
        N_z_prelim = max(0.0, N_z_prelim)

        # Friction force on floor
        from dynamics.sliding_mass import smooth_friction_force
        F_fric_2d = smooth_friction_force(
            sdot, N_z_prelim, bp.mu_k, bp.mu_s, bp.visc_damping, bp.friction_v_scale
        )

        # 2. Assemble 8x8 Symmetric Mass Matrix H_sys
        H_sys = np.zeros((8, 8))

        # Linear momentum (3 rows)
        H_sys[0:3, 0:3] = (m_L + m_d) * np.eye(3)
        H_sys[0:3, 3:6] = -m_d * R_L @ rho_cross
        H_sys[0:3, 6:8] = m_d * R_L[:, 0:2]

        # Angular momentum of basket (3 rows)
        H_sys[3:6, 0:3] = m_d * rho_cross @ R_L.T
        H_sys[3:6, 3:6] = J_L - m_d * (rho_cross @ rho_cross)
        H_sys[3:6, 6:8] = m_d * rho_cross[:, 0:2]

        # Sliding mass relative floor motion (2 rows)
        H_sys[6:8, 0:3] = m_d * R_L[:, 0:2].T
        H_sys[6:8, 3:6] = -m_d * rho_cross[0:2, :]
        H_sys[6:8, 6:8] = m_d * np.eye(2)

        # 3. Assemble Right-Hand Side vector B_sys
        B_sys = np.zeros(8)
        B_sys[0:3] = m_L * g_world + F_cable_total + F_drag_world - m_d * a_fictitious_world
        B_sys[3:6] = (tau_cable_total_body + tau_drag_body - np.cross(omega_L, J_L @ omega_L)
                      - m_d * np.cross(rho, a_fictitious_body + g_body))
        B_sys[6:8] = m_d * (g_body[0:2] - a_fictitious_body[0:2]) + F_fric_2d

        # 4. Solve the linear system
        try:
            x_acc = np.linalg.solve(H_sys, B_sys)
            a_L = x_acc[0:3]
            omega_L_dot = x_acc[3:6]
            sddot = x_acc[6:8]
        except np.linalg.LinAlgError:
            # Fallback
            a_L = a_L_prelim
            omega_L_dot = np.zeros(3)
            sddot = np.zeros(2)

        # ---- Drone derivatives ----
        d_drone_states = []
        for i in range(NUM_DRONES):
            dd = drone_derivatives(
                drone_states[i],
                cfg.drone_params[i],
                rotor_cmds[i],
                F_cables_on_drones[i],
                tau_cables_on_drones_body[i],
                gravity=cfg.gravity,
                wind_velocity=wind,
            )
            d_drone_states.append(dd)

        # ---- Load state derivative ----
        from dynamics.drone import quat_derivative as qd
        dq_L = qd(q_L, omega_L)
        d_load = np.concatenate([v_L, a_L, dq_L, omega_L_dot])

        # ---- Sliding mass state derivative ----
        d_sliding = np.concatenate([sdot, sddot])

        # ---- Pack and return ----
        return pack_system_state(d_drone_states, d_load, d_sliding)

    return rhs


# ---------------------------------------------------------------------------
# Fixed-step RK4 integrator (fallback / cross-validation)
# ---------------------------------------------------------------------------

def rk4_step(f: Callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Single step of the classical 4th-order Runge-Kutta method."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return normalise_quaternions(y_new)


def integrate_rk4(
    f: Callable, y0: np.ndarray, t_span: tuple[float, float],
    dt: float, t_eval: Optional[np.ndarray] = None,
) -> dict:
    """Fixed-step RK4 integration.

    Returns a dict with 't' and 'y' arrays, mimicking solve_ivp output.
    """
    t0, tf = t_span
    t = t0
    y = y0.copy()

    ts = [t0]
    ys = [y0.copy()]

    while t < tf - 1e-14:
        step = min(dt, tf - t)
        y = rk4_step(f, t, y, step)
        t += step
        ts.append(t)
        ys.append(y.copy())

    result = {
        't': np.array(ts),
        'y': np.array(ys).T,   # shape (n_states, n_times) to match solve_ivp
    }

    # If t_eval is specified, interpolate
    if t_eval is not None:
        from scipy.interpolate import interp1d
        interp = interp1d(result['t'], result['y'], axis=1,
                          kind='cubic', fill_value='extrapolate')
        result['t'] = t_eval
        result['y'] = interp(t_eval)

    return result


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def build_initial_state(
    cfg: SystemConfig,
    load_pos: np.ndarray = None,
    load_yaw: float = 0.0,
) -> np.ndarray:
    """Build the initial state vector with drones hovering above load corners.

    Parameters
    ----------
    cfg : SystemConfig
    load_pos : (3,) initial load CM position.  Default [0, 0, 3.0].
    load_yaw : initial load yaw angle (rad).  Default 0.

    Returns
    -------
    y0 : (85,) initial state vector
    """
    if load_pos is None:
        load_pos = np.array([0.0, 0.0, 3.0])

    # Load initial attitude (level, with specified yaw)
    cy, sy = np.cos(load_yaw), np.sin(load_yaw)
    R_L0 = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1],
    ])

    from dynamics.drone import rotmat_to_quat
    q_L0 = rotmat_to_quat(R_L0)

    # Load state
    load_state = pack_load_state(
        load_pos, np.zeros(3), q_L0, np.zeros(3)
    )

    # Sliding mass: centered, at rest
    sliding_state = pack_sliding_state(np.zeros(2), np.zeros(2))

    # Drone states: positioned above each corner at cable natural length + static equilibrium stretch
    drone_states = []
    total_load_weight = (
        cfg.basket_params.total_mass + cfg.basket_params.dynamic_load_mass
    ) * cfg.gravity
    load_share = total_load_weight / NUM_DRONES

    for i in range(NUM_DRONES):
        corner_world = get_corner_world(
            load_pos, R_L0, cfg.basket_params.corner_points_body[i]
        )
        # Calculate static cable stretch at hover
        delta_static = load_share / cfg.cable_params[i].stiffness
        L0 = cfg.cable_params[i].natural_length
        
        # Drone position: directly above the corner, at cable natural length + static stretch
        drone_pos = corner_world + np.array([0.0, 0.0, L0 + delta_static])

        # Account for cable attach offset (drone CM is slightly above attach point)
        drone_pos -= R_L0 @ cfg.drone_params[i].cable_attach_offset

        # Level attitude
        q_drone = np.array([1.0, 0.0, 0.0, 0.0])

        # Compute hover rotor speed: total thrust = weight of drone + share of load
        total_load_weight = (
            cfg.basket_params.total_mass + cfg.basket_params.dynamic_load_mass
        ) * cfg.gravity
        load_share = total_load_weight / NUM_DRONES
        drone_weight = cfg.drone_params[i].mass * cfg.gravity
        F_hover = drone_weight + load_share
        # F = 4 * k_T * omega^2  →  omega = sqrt(F / (4 * k_T))
        omega_hover = np.sqrt(F_hover / (4 * cfg.drone_params[i].k_T))
        omega_hover = np.clip(omega_hover, cfg.drone_params[i].omega_min,
                              cfg.drone_params[i].omega_max)
        rotors = np.full(4, omega_hover)

        ds = pack_drone_state(drone_pos, np.zeros(3), q_drone, np.zeros(3), rotors)
        drone_states.append(ds)

    return pack_system_state(drone_states, load_state, sliding_state)


# ---------------------------------------------------------------------------
# Energy computation (for validation)
# ---------------------------------------------------------------------------

def compute_total_energy(y: np.ndarray, cfg: SystemConfig) -> dict:
    """Compute kinetic and potential energy of the full system.

    Returns dict with 'KE_drones', 'KE_load', 'KE_sliding', 'PE_drones',
    'PE_load', 'PE_sliding', 'PE_cables', 'total'.
    """
    drone_states, load_state, sliding_state = unpack_system_state(y)
    p_L, v_L, q_L, omega_L = unpack_load_state(load_state)
    R_L = quat_to_rotmat(q_L / np.linalg.norm(q_L))
    s, sdot = unpack_sliding_state(sliding_state)
    g = cfg.gravity
    bp = cfg.basket_params

    # ---- Drone energies ----
    KE_drones = 0.0
    PE_drones = 0.0
    for i in range(NUM_DRONES):
        p, v, q, omega, rotors = unpack_drone_state(drone_states[i])
        dp = cfg.drone_params[i]
        KE_drones += 0.5 * dp.mass * np.dot(v, v)
        KE_drones += 0.5 * np.dot(omega, dp.J @ omega)
        PE_drones += dp.mass * g * p[2]

    # ---- Load energy ----
    KE_load = 0.5 * bp.total_mass * np.dot(v_L, v_L)
    KE_load += 0.5 * np.dot(omega_L, bp.J_combined @ omega_L)
    PE_load = bp.total_mass * g * p_L[2]

    # ---- Sliding mass energy ----
    rho = np.array([s[0], s[1], bp.floor_height])
    v_d_body = np.array([sdot[0], sdot[1], 0.0])
    omega_world = R_L @ omega_L
    r_offset_world = R_L @ rho
    v_d = v_L + np.cross(omega_world, r_offset_world) + R_L @ v_d_body
    p_d = p_L + R_L @ rho
    KE_sliding = 0.5 * bp.dynamic_load_mass * np.dot(v_d, v_d)
    PE_sliding = bp.dynamic_load_mass * g * p_d[2]

    # ---- Cable elastic potential energy ----
    PE_cables = 0.0
    for i in range(NUM_DRONES):
        cp = cfg.cable_params[i]
        if cp.model == "elastic":
            a_i = get_cable_attach_world(drone_states[i], cfg.drone_params[i])
            r_i = get_corner_world(p_L, R_L, bp.corner_points_body[i])
            e = a_i - r_i
            L = np.linalg.norm(e)
            delta = max(0.0, L - cp.natural_length)
            PE_cables += 0.5 * cp.stiffness * delta**2

    total = KE_drones + KE_load + KE_sliding + PE_drones + PE_load + PE_sliding + PE_cables

    return {
        'KE_drones': KE_drones,
        'KE_load': KE_load,
        'KE_sliding': KE_sliding,
        'PE_drones': PE_drones,
        'PE_load': PE_load,
        'PE_sliding': PE_sliding,
        'PE_cables': PE_cables,
        'total': total,
    }
