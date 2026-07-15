"""
sim/run_sim.py — Main entry point for the cooperative 4-UAV slung load simulation.

Usage:
    python -m sim.run_sim                           # run with defaults
    python -m sim.run_sim --config sim/config.yaml  # specify config file
    python -m sim.run_sim --scenario circle         # override trajectory
    python -m sim.run_sim --integrator RK4          # use fixed-step RK4
    python -m sim.run_sim --no-animate              # skip animation, plots only
"""

from __future__ import annotations

import sys
import os
import argparse
import time
import numpy as np
import yaml

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dynamics.drone import DroneParams
from dynamics.cable import CableParams
from dynamics.load import BasketParams
from dynamics.system import (
    SystemConfig, TOTAL_STATE_DIM,
    make_system_rhs, build_initial_state, integrate_rk4,
)
from control.formation_mode import make_controller
from viz.plots import compute_telemetry, plot_telemetry
from viz.animate_3d import animate_simulation


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_system_config(raw: dict) -> SystemConfig:
    """Convert raw YAML dict to SystemConfig dataclass."""
    d = raw.get('drone', {})
    c = raw.get('cable', {})
    b = raw.get('basket', {})
    s = raw.get('sim', {})
    w = raw.get('wind', {})

    drone_params = [
        DroneParams(
            mass=d.get('mass_kg', 2.0),
            J=np.diag(d.get('inertia_kgm2', [0.02, 0.02, 0.04])),
            arm_length=d.get('arm_length_m', 0.25),
            k_T=d.get('k_T', 1.5e-5),
            k_Q=d.get('k_Q', 2.5e-7),
            tau_motor=d.get('motor_time_constant_s', 0.03),
            omega_max=d.get('max_motor_speed_rad_s', 900.0),
            omega_min=d.get('min_motor_speed_rad_s', 50.0),
            cable_attach_offset=np.array(d.get('cable_attach_offset_m', [0, 0, -0.05])),
            drag_coeff=d.get('drag_coeff', 0.15),
        )
        for _ in range(4)
    ]

    cable_params = [
        CableParams(
            natural_length=c.get('natural_length_m', 1.2),
            stiffness=c.get('stiffness_N_per_m', 1500.0),
            damping=c.get('damping_Ns_per_m', 15.0),
            model=c.get('model', 'elastic'),
            baumgarte_alpha=c.get('baumgarte_alpha', 10.0),
            baumgarte_beta=c.get('baumgarte_beta', 10.0),
        )
        for _ in range(4)
    ]

    basket_params = BasketParams(
        empty_mass=b.get('empty_mass_kg', 1.2),
        half_extent=np.array(b.get('half_extent_m', [0.3, 0.3])),
        corner_height=b.get('corner_height_m', 0.15),
        static_load_mass=b.get('static_load_kg', 1.5),
        static_load_pos_body=np.array(b.get('static_load_pos_body_m', [0, 0, -0.05])),
        dynamic_load_mass=b.get('dynamic_load_kg', 0.8),
        floor_height=b.get('dynamic_load_floor_height_m', -0.10),
        mu_k=b.get('friction_mu_kinetic', 0.3),
        mu_s=b.get('friction_mu_static', 0.4),
        visc_damping=b.get('floor_viscous_damping', 0.5),
        friction_v_scale=b.get('friction_v_scale', 0.01),
    )

    cfg = SystemConfig(
        drone_params=drone_params,
        cable_params=cable_params,
        basket_params=basket_params,
        gravity=s.get('gravity', 9.80665),
        wind_enabled=w.get('enabled', False),
        wind_mean_velocity=np.array(w.get('mean_velocity', [0, 0, 0])),
        wind_gust_amplitude=w.get('gust_amplitude', 0.0),
        wind_gust_frequency=w.get('gust_frequency', 0.0),
    )

    return cfg


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(
    config_path: str = None,
    scenario: str = None,
    integrator: str = None,
    total_time: float = None,
    animate: bool = True,
    save_animation: str = None,
    save_plots: str = None,
) -> dict:
    """Run the full simulation.

    Parameters
    ----------
    config_path : path to YAML config (default: sim/config.yaml)
    scenario : override trajectory type ('hover', 'circle', 'lemniscate', 'step')
    integrator : override integrator ('Radau', 'BDF', 'RK45', 'RK4')
    total_time : override simulation duration (s)
    animate : whether to show 3D animation
    save_animation : path to save animation file
    save_plots : path to save telemetry plots

    Returns
    -------
    dict with 't', 'y', 'telemetry', 'cfg'
    """
    # ---- Load config ----
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, 'sim', 'config.yaml')
    raw_cfg = load_config(config_path)
    cfg = build_system_config(raw_cfg)

    sim_cfg = raw_cfg.get('sim', {})
    ctrl_cfg = raw_cfg.get('control', {})
    traj_cfg = raw_cfg.get('trajectory', {})

    # Apply overrides
    if scenario:
        traj_cfg['type'] = scenario
    if total_time:
        sim_cfg['total_time_s'] = total_time
    if integrator:
        sim_cfg['integrator'] = integrator

    T_total = sim_cfg.get('total_time_s', 30.0)
    dt_rk4 = sim_cfg.get('dt_fixed_fallback', 1e-4)
    integ_method = sim_cfg.get('integrator', 'Radau')
    rtol = sim_cfg.get('rtol', 1e-6)
    atol = sim_cfg.get('atol', 1e-8)
    max_step = sim_cfg.get('max_step', 0.01)
    output_rate = sim_cfg.get('output_rate_hz', 100.0)

    # ---- Build controller ----
    controller = make_controller(ctrl_cfg, traj_cfg)

    # ---- Build initial state ----
    hover_pos = np.array(traj_cfg.get('hover_pos', [0.0, 0.0, 3.0]))
    y0 = build_initial_state(cfg, load_pos=hover_pos)
    assert len(y0) == TOTAL_STATE_DIM, f"State dim mismatch: {len(y0)} != {TOTAL_STATE_DIM}"

    # ---- Build RHS ----
    rhs = make_system_rhs(cfg, controller)

    # ---- Time evaluation points ----
    N_out = int(T_total * output_rate)
    t_eval = np.linspace(0, T_total, N_out + 1)

    # ---- Integrate ----
    print(f"\n{'='*70}")
    print(f"  4-UAV Cooperative Slung Load Simulation")
    print(f"  Integrator: {integ_method}  |  Duration: {T_total:.1f} s")
    print(f"  Control mode: {ctrl_cfg.get('mode', 'formation')}")
    print(f"  Trajectory: {traj_cfg.get('type', 'hover')}")
    print(f"  Cable model: {cfg.cable_params[0].model}")
    print(f"  State dimension: {TOTAL_STATE_DIM}")
    print(f"{'='*70}\n")

    t_start = time.time()

    if integ_method.upper() == 'RK4':
        print(f"Using fixed-step RK4 (dt = {dt_rk4:.1e} s) ...")
        result = integrate_rk4(rhs, y0, (0, T_total), dt_rk4, t_eval=t_eval)
        t_arr = result['t']
        y_arr = result['y']
    else:
        from scipy.integrate import solve_ivp

        print(f"Using scipy.integrate.solve_ivp(method='{integ_method}') ...")
        print(f"  rtol={rtol:.0e}, atol={atol:.0e}, max_step={max_step}")
        print(f"  Integrating ... ", end='', flush=True)

        sol = solve_ivp(
            rhs, [0, T_total], y0,
            method=integ_method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=False,
        )

        if not sol.success:
            print(f"\n  WARNING: Integration failed: {sol.message}")
        else:
            print(f"done.")

        t_arr = sol.t
        y_arr = sol.y

    elapsed = time.time() - t_start
    print(f"\n  Integration completed in {elapsed:.1f} s")
    print(f"  {len(t_arr)} output points")

    # ---- Compute telemetry ----
    print("  Computing telemetry ...", end=' ', flush=True)
    telemetry = compute_telemetry(t_arr, y_arr, cfg)
    print("done.")

    # ---- Print summary ----
    print(f"\n  Final tensions: {telemetry['tensions'][:, -1].round(2)} N")
    print(f"  Final load pos: {telemetry['load_pos'][:, -1].round(3)} m")
    print(f"  Sliding mass final pos: {telemetry['sliding_pos'][:, -1].round(4)} m")
    print(f"  Energy initial: {telemetry['energies'][0]:.2f} J")
    print(f"  Energy final:   {telemetry['energies'][-1]:.2f} J")
    E_change = telemetry['energies'][-1] - telemetry['energies'][0]
    print(f"  Energy change:  {E_change:.2f} J ({E_change/max(abs(telemetry['energies'][0]),1e-6)*100:.2f} %)")

    # ---- Plots ----
    plot_telemetry(t_arr, telemetry, save_path=save_plots)

    # ---- Animation ----
    anim = None
    if animate:
        anim = animate_simulation(t_arr, y_arr, cfg, skip=max(1, len(t_arr)//500),
                                  save_path=save_animation)

    # Open all Matplotlib windows simultaneously
    import matplotlib.pyplot as plt
    print("Opening telemetry plots and 3D animation windows...")
    plt.show()

    return {
        't': t_arr,
        'y': y_arr,
        'telemetry': telemetry,
        'cfg': cfg,
        'anim': anim,  # Keep reference alive to prevent garbage collection
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cooperative 4-UAV Cable-Suspended Load Simulation'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['hover', 'circle', 'lemniscate', 'step'],
                        help='Override trajectory type')
    parser.add_argument('--integrator', type=str, default=None,
                        choices=['Radau', 'BDF', 'RK45', 'RK4'],
                        help='Override integrator method')
    parser.add_argument('--time', type=float, default=None,
                        help='Override total simulation time (s)')
    parser.add_argument('--no-animate', action='store_true',
                        help='Skip 3D animation (plots only)')
    parser.add_argument('--save-animation', type=str, default=None,
                        help='Save animation to file (e.g. sim.mp4)')
    parser.add_argument('--save-plots', type=str, default=None,
                        help='Save telemetry plots to file (e.g. plots.png)')

    args = parser.parse_args()

    run_simulation(
        config_path=args.config,
        scenario=args.scenario,
        integrator=args.integrator,
        total_time=args.time,
        animate=not args.no_animate,
        save_animation=args.save_animation,
        save_plots=args.save_plots,
    )


if __name__ == '__main__':
    main()
