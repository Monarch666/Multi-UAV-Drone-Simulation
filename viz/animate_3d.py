"""
viz/animate_3d.py — 3D animation of the multi-UAV slung load system.

Renders an animated 3D view showing:
  • 4 quadrotor UAVs (drawn as X-frame arms + rotors)
  • 4 cables (lines colored by tension magnitude)
  • Rectangular basket (wireframe)
  • Sliding mass (marker with trail)
  • Static load position (marker)
  • Ground plane grid

Uses matplotlib's FuncAnimation for lightweight, dependency-free animation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from dynamics.system import unpack_system_state, NUM_DRONES
from dynamics.drone import (
    unpack_drone_state, quat_to_rotmat,
    get_cable_attach_world, DroneParams,
)
from dynamics.load import (
    unpack_load_state, get_corner_world, BasketParams,
)
from dynamics.sliding_mass import unpack_sliding_state


def draw_drone(ax, pos, R, arm_length, color='blue', alpha=0.8):
    """Draw a quadrotor as an X-frame with rotor discs."""
    L = arm_length
    # Arm endpoints in body frame (X-config)
    arms_body = np.array([
        [ L,  L, 0],
        [-L, -L, 0],
        [ L, -L, 0],
        [-L,  L, 0],
    ]) / np.sqrt(2)

    arms_world = (R @ arms_body.T).T + pos

    # Draw two arms
    ax.plot3D(
        [arms_world[0, 0], arms_world[1, 0]],
        [arms_world[0, 1], arms_world[1, 1]],
        [arms_world[0, 2], arms_world[1, 2]],
        color=color, linewidth=2.5, alpha=alpha
    )
    ax.plot3D(
        [arms_world[2, 0], arms_world[3, 0]],
        [arms_world[2, 1], arms_world[3, 1]],
        [arms_world[2, 2], arms_world[3, 2]],
        color=color, linewidth=2.5, alpha=alpha
    )

    # Rotor discs (small circles at arm ends)
    for arm_end in arms_world:
        ax.scatter(*arm_end, s=40, color=color, alpha=alpha, zorder=5)

    # Center marker
    ax.scatter(*pos, s=20, color='black', alpha=0.9, zorder=6)


def draw_basket(ax, p_L, R_L, bp: BasketParams, alpha=0.4):
    """Draw the basket as a wireframe rectangular box."""
    ax2, ay2 = bp.half_extent
    # Basket top and bottom in body frame
    hz_top = bp.corner_height
    hz_bot = bp.floor_height

    # 8 corners of the basket box
    corners_body = np.array([
        [ ax2,  ay2, hz_top],
        [-ax2,  ay2, hz_top],
        [-ax2, -ay2, hz_top],
        [ ax2, -ay2, hz_top],
        [ ax2,  ay2, hz_bot],
        [-ax2,  ay2, hz_bot],
        [-ax2, -ay2, hz_bot],
        [ ax2, -ay2, hz_bot],
    ]) - bp.cm_shift

    corners_world = (R_L @ corners_body.T).T + p_L

    # Draw edges
    # Top face
    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            color='saddlebrown', linewidth=1.5, alpha=0.7
        )
    # Bottom face
    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D(
            [corners_world[i+4, 0], corners_world[j+4, 0]],
            [corners_world[i+4, 1], corners_world[j+4, 1]],
            [corners_world[i+4, 2], corners_world[j+4, 2]],
            color='saddlebrown', linewidth=1.5, alpha=0.7
        )
    # Vertical edges
    for i in range(4):
        ax.plot3D(
            [corners_world[i, 0], corners_world[i+4, 0]],
            [corners_world[i, 1], corners_world[i+4, 1]],
            [corners_world[i, 2], corners_world[i+4, 2]],
            color='saddlebrown', linewidth=1.0, alpha=0.5
        )

    # Floor face (semi-transparent)
    floor_verts = [list(corners_world[4:8])]
    floor_poly = Poly3DCollection(floor_verts, alpha=0.15, facecolor='burlywood',
                                   edgecolor='saddlebrown', linewidth=0.5)
    ax.add_collection3d(floor_poly)


def animate_simulation(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    cfg,
    skip: int = 5,
    save_path: str = None,
    figsize: tuple = (14, 10),
):
    """Create a 3D animation of the simulation.

    Parameters
    ----------
    t_arr : (N,) time array
    y_arr : (n_states, N) state array
    cfg : SystemConfig
    skip : frame skip factor (1 = every frame, 5 = every 5th frame)
    save_path : optional path to save animation (e.g. 'sim.mp4')
    figsize : figure size
    """
    drone_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    N = len(t_arr)
    frame_indices = list(range(0, N, skip))

    # Pre-compute bounds
    all_drone_pos = []
    all_load_pos = []
    for k in frame_indices:
        y = y_arr[:, k]
        drone_states, load_state, sliding_state = unpack_system_state(y)
        p_L, _, _, _ = unpack_load_state(load_state)
        all_load_pos.append(p_L)
        for i in range(4):
            p_i, _, _, _, _ = unpack_drone_state(drone_states[i])
            all_drone_pos.append(p_i)

    all_pos = np.array(all_drone_pos + all_load_pos)
    margin = 1.0
    x_lim = [all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin]
    y_lim = [all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin]
    z_lim = [max(0, all_pos[:, 2].min() - margin), all_pos[:, 2].max() + margin]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Sliding mass trail storage
    sliding_trail_x = []
    sliding_trail_y = []
    sliding_trail_z = []

    def update(frame_idx):
        ax.cla()
        k = frame_indices[frame_idx]
        y = y_arr[:, k]
        drone_states, load_state, sliding_state = unpack_system_state(y)
        p_L, v_L, q_L, omega_L = unpack_load_state(load_state)
        q_L = q_L / np.linalg.norm(q_L)
        R_L = quat_to_rotmat(q_L)
        s, sdot = unpack_sliding_state(sliding_state)
        bp = cfg.basket_params

        # Draw ground grid
        gx = np.linspace(x_lim[0], x_lim[1], 10)
        gy = np.linspace(y_lim[0], y_lim[1], 10)
        GX, GY = np.meshgrid(gx, gy)
        GZ = np.zeros_like(GX)
        ax.plot_surface(GX, GY, GZ, alpha=0.05, color='gray')

        # Draw basket
        draw_basket(ax, p_L, R_L, bp)

        # Draw cables and drones
        tension_norm = Normalize(vmin=0, vmax=30)
        # Use colormaps directly to support modern Matplotlib (v3.9+)
        cmap = plt.colormaps['RdYlGn_r']

        for i in range(NUM_DRONES):
            dp = cfg.drone_params[i]
            ds = drone_states[i]
            p_i, v_i, q_i, omega_i, rots = unpack_drone_state(ds)
            q_i = q_i / np.linalg.norm(q_i)
            R_i = quat_to_rotmat(q_i)

            # Draw drone
            draw_drone(ax, p_i, R_i, dp.arm_length, color=drone_colors[i])

            # Draw cable
            a_i = get_cable_attach_world(ds, dp)
            r_i = get_corner_world(p_L, R_L, bp.corner_points_body[i])

            # Color by tension (approximate)
            e = a_i - r_i
            L = np.linalg.norm(e)
            delta = max(0.0, L - cfg.cable_params[i].natural_length)
            T_approx = cfg.cable_params[i].stiffness * delta
            cable_color = cmap(tension_norm(T_approx))

            ax.plot3D(
                [a_i[0], r_i[0]],
                [a_i[1], r_i[1]],
                [a_i[2], r_i[2]],
                color=cable_color, linewidth=2.0, alpha=0.9
            )

        # Draw sliding mass
        rho = np.array([s[0], s[1], bp.floor_height])
        p_d = p_L + R_L @ (rho - bp.cm_shift)
        sliding_trail_x.append(p_d[0])
        sliding_trail_y.append(p_d[1])
        sliding_trail_z.append(p_d[2])

        ax.scatter(*p_d, s=80, color='red', marker='o', zorder=10, label='Dynamic load')
        # Trail
        trail_len = min(len(sliding_trail_x), 100)
        ax.plot3D(
            sliding_trail_x[-trail_len:],
            sliding_trail_y[-trail_len:],
            sliding_trail_z[-trail_len:],
            color='red', linewidth=0.5, alpha=0.3
        )

        # Static load marker
        q_s = bp.static_load_pos_combined
        p_s = p_L + R_L @ q_s
        ax.scatter(*p_s, s=60, color='purple', marker='s', zorder=10, label='Static load')

        # Labels and limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel('X (East) [m]')
        ax.set_ylabel('Y (North) [m]')
        ax.set_zlabel('Z (Up) [m]')
        ax.set_title(f'4-UAV Slung Load Simulation  t = {t_arr[k]:.2f} s',
                     fontsize=12, fontweight='bold')

        # Legend (only first frame items)
        if frame_idx == 0:
            for i in range(4):
                ax.scatter([], [], [], color=drone_colors[i], s=30,
                           label=f'Drone {i+1}')
            ax.legend(loc='upper left', fontsize=7)

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=50, blit=False, repeat=True)

    if save_path:
        print(f"Saving animation to {save_path} ...")
        anim.save(save_path, writer='ffmpeg', fps=20, dpi=100)
        print("Done.")

    return anim
