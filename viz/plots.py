"""
viz/plots.py — Telemetry plotting for simulation results.

Generates publication-quality plots of:
  • Cable tensions over time
  • Tracking errors (position, attitude)
  • Total energy conservation check
  • Sliding mass position trajectory
  • Rotor speeds
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dynamics.system import (
    unpack_system_state, compute_total_energy,
    SystemConfig, NUM_DRONES,
)
from dynamics.drone import unpack_drone_state, quat_to_rotmat
from dynamics.load import unpack_load_state, get_corner_world
from dynamics.cable import compute_cable_force_elastic
from dynamics.sliding_mass import unpack_sliding_state


def compute_telemetry(t_arr: np.ndarray, y_arr: np.ndarray,
                      cfg: SystemConfig) -> dict:
    """Extract telemetry from simulation results.

    Parameters
    ----------
    t_arr : (N,) time array
    y_arr : (n_states, N) state array (solve_ivp format)
    cfg : SystemConfig

    Returns
    -------
    dict of telemetry arrays
    """
    N = len(t_arr)

    # Pre-allocate
    tensions = np.zeros((4, N))
    drone_pos = np.zeros((4, 3, N))
    drone_vel = np.zeros((4, 3, N))
    rotor_speeds = np.zeros((4, 4, N))
    load_pos = np.zeros((3, N))
    load_vel = np.zeros((3, N))
    sliding_pos = np.zeros((2, N))
    sliding_vel = np.zeros((2, N))
    energies = np.zeros(N)

    for k in range(N):
        y = y_arr[:, k]
        drone_states, load_state, sliding_state = unpack_system_state(y)
        p_L, v_L, q_L, omega_L = unpack_load_state(load_state)
        q_L = q_L / np.linalg.norm(q_L)
        R_L = quat_to_rotmat(q_L)
        s, sdot = unpack_sliding_state(sliding_state)

        load_pos[:, k] = p_L
        load_vel[:, k] = v_L
        sliding_pos[:, k] = s
        sliding_vel[:, k] = sdot

        for i in range(4):
            p_i, v_i, q_i, omega_i, rots = unpack_drone_state(drone_states[i])
            drone_pos[i, :, k] = p_i
            drone_vel[i, :, k] = v_i
            rotor_speeds[i, :, k] = rots

            # Compute cable tension
            from dynamics.drone import (
                get_cable_attach_world, get_cable_attach_velocity_world,
            )
            from dynamics.load import get_corner_world, get_corner_velocity_world

            a_i = get_cable_attach_world(drone_states[i], cfg.drone_params[i])
            r_i = get_corner_world(p_L, R_L, cfg.basket_params.corner_points_body[i])
            va_i = get_cable_attach_velocity_world(drone_states[i], cfg.drone_params[i])
            vr_i = get_corner_velocity_world(
                v_L, omega_L, R_L, cfg.basket_params.corner_points_body[i]
            )
            T, _, _ = compute_cable_force_elastic(a_i, r_i, va_i, vr_i, cfg.cable_params[i])
            tensions[i, k] = T

        energy = compute_total_energy(y, cfg)
        energies[k] = energy['total']

    return {
        'tensions': tensions,
        'drone_pos': drone_pos,
        'drone_vel': drone_vel,
        'rotor_speeds': rotor_speeds,
        'load_pos': load_pos,
        'load_vel': load_vel,
        'sliding_pos': sliding_pos,
        'sliding_vel': sliding_vel,
        'energies': energies,
    }


def plot_telemetry(t_arr: np.ndarray, telemetry: dict,
                   save_path: str = None):
    """Generate a multi-panel telemetry figure.

    Parameters
    ----------
    t_arr : (N,) time array
    telemetry : dict from compute_telemetry
    save_path : optional path to save figure
    """
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    drone_labels = [f'Drone {i+1}' for i in range(4)]

    # ---- 1. Cable tensions ----
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(4):
        ax1.plot(t_arr, telemetry['tensions'][i], color=colors[i],
                 label=drone_labels[i], linewidth=1.2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tension (N)')
    ax1.set_title('Cable Tensions')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- 2. Load position ----
    ax2 = fig.add_subplot(gs[0, 1])
    labels_xyz = ['x', 'y', 'z']
    colors_xyz = ['#e74c3c', '#3498db', '#2ecc71']
    for j in range(3):
        ax2.plot(t_arr, telemetry['load_pos'][j], color=colors_xyz[j],
                 label=labels_xyz[j], linewidth=1.2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Load CM Position')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- 3. Load velocity ----
    ax3 = fig.add_subplot(gs[0, 2])
    for j in range(3):
        ax3.plot(t_arr, telemetry['load_vel'][j], color=colors_xyz[j],
                 label=labels_xyz[j], linewidth=1.2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Load CM Velocity')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- 4. Drone positions (z-component) ----
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(4):
        ax4.plot(t_arr, telemetry['drone_pos'][i, 2, :], color=colors[i],
                 label=drone_labels[i], linewidth=1.2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Altitude (m)')
    ax4.set_title('Drone Altitudes')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ---- 5. Rotor speeds (drone 1) ----
    ax5 = fig.add_subplot(gs[1, 1])
    rotor_colors = ['#8e44ad', '#d35400', '#16a085', '#c0392b']
    for j in range(4):
        ax5.plot(t_arr, telemetry['rotor_speeds'][0, j, :],
                 color=rotor_colors[j], label=f'Rotor {j+1}', linewidth=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (rad/s)')
    ax5.set_title('Drone 1 Rotor Speeds')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ---- 6. Energy ----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t_arr, telemetry['energies'], color='#2c3e50', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Total Energy (J)')
    ax6.set_title('Total System Energy')
    ax6.grid(True, alpha=0.3)

    # ---- 7. Sliding mass position ----
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t_arr, telemetry['sliding_pos'][0], color='#e74c3c',
             label='s_x', linewidth=1.2)
    ax7.plot(t_arr, telemetry['sliding_pos'][1], color='#3498db',
             label='s_y', linewidth=1.2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Position (m)')
    ax7.set_title('Sliding Mass Position')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # ---- 8. Sliding mass trajectory (top view) ----
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(telemetry['sliding_pos'][0], telemetry['sliding_pos'][1],
             color='#e74c3c', linewidth=0.8, alpha=0.7)
    ax8.scatter(telemetry['sliding_pos'][0, 0], telemetry['sliding_pos'][1, 0],
                color='green', s=50, zorder=5, label='Start')
    ax8.scatter(telemetry['sliding_pos'][0, -1], telemetry['sliding_pos'][1, -1],
                color='red', s=50, zorder=5, label='End')
    # Draw basket boundary
    ax, ay = 0.3, 0.3  # half-extents
    rect_x = [ax, -ax, -ax, ax, ax]
    rect_y = [ay, ay, -ay, -ay, ay]
    ax8.plot(rect_x, rect_y, 'k--', linewidth=1, alpha=0.5, label='Basket boundary')
    ax8.set_xlabel('s_x (m)')
    ax8.set_ylabel('s_y (m)')
    ax8.set_title('Sliding Mass Trajectory (Top View)')
    ax8.set_aspect('equal')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    # ---- 9. Drone XY positions (top view) ----
    ax9 = fig.add_subplot(gs[2, 2])
    for i in range(4):
        ax9.plot(telemetry['drone_pos'][i, 0, :],
                 telemetry['drone_pos'][i, 1, :],
                 color=colors[i], label=drone_labels[i], linewidth=0.8, alpha=0.7)
        ax9.scatter(telemetry['drone_pos'][i, 0, 0],
                    telemetry['drone_pos'][i, 1, 0],
                    color=colors[i], s=30, zorder=5)
    ax9.plot(telemetry['load_pos'][0], telemetry['load_pos'][1],
             'k-', linewidth=1.5, label='Load CM')
    ax9.set_xlabel('x (m)')
    ax9.set_ylabel('y (m)')
    ax9.set_title('Drone & Load XY Trajectories')
    ax9.set_aspect('equal')
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3)

    plt.suptitle('Multi-UAV Slung Load Simulation Telemetry', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Telemetry plot saved to: {save_path}")

