"""
control/mixer.py — Rotor speed ↔ thrust/torque mapping for X-frame quadrotors.

Provides the mixer matrix M and its inverse M_inv:
    [F_total, τ_x, τ_y, τ_z]ᵀ = M · [ω₁², ω₂², ω₃², ω₄²]ᵀ
    [ω₁², …, ω₄²]ᵀ = M_inv · [F_cmd, τ_x_cmd, τ_y_cmd, τ_z_cmd]ᵀ

Rotor layout (top-view, body frame x-forward, y-left):
    Rotor 1: front-right (+x, −y) — CW  → +yaw torque
    Rotor 2: front-left  (+x, +y) — CCW → −yaw torque
    Rotor 3: rear-left   (−x, +y) — CW  → +yaw torque
    Rotor 4: rear-right  (−x, −y) — CCW → −yaw torque
"""

from __future__ import annotations

import numpy as np


def build_mixer_matrix(arm_length: float, k_T: float, k_Q: float) -> np.ndarray:
    """Build the 4×4 mixer matrix.

    Returns M such that wrench = M @ omega_sq.
    """
    L = arm_length / np.sqrt(2)   # effective arm for X-frame
    return np.array([
        [ k_T,     k_T,     k_T,     k_T    ],
        [-k_T*L,   k_T*L,   k_T*L,  -k_T*L  ],
        [ k_T*L,   k_T*L,  -k_T*L,  -k_T*L  ],
        [ k_Q,    -k_Q,     k_Q,    -k_Q     ],
    ])


def build_mixer_inverse(arm_length: float, k_T: float, k_Q: float) -> np.ndarray:
    """Inverse mixer: wrench command → rotor speed² commands."""
    M = build_mixer_matrix(arm_length, k_T, k_Q)
    return np.linalg.inv(M)


def wrench_to_rotor_speeds(
    F_cmd: float,
    tau_cmd: np.ndarray,
    arm_length: float,
    k_T: float,
    k_Q: float,
    omega_min: float,
    omega_max: float,
) -> np.ndarray:
    """Convert thrust + torque command to individual rotor speed commands.

    Parameters
    ----------
    F_cmd   : total thrust command (N)
    tau_cmd : (3,) body torque command [τ_x, τ_y, τ_z] (N·m)
    arm_length, k_T, k_Q : drone parameters
    omega_min, omega_max : rotor speed limits (rad/s)

    Returns
    -------
    omega_cmd : (4,) rotor speed commands (rad/s), clamped to [omega_min, omega_max]
    """
    M_inv = build_mixer_inverse(arm_length, k_T, k_Q)
    wrench = np.array([F_cmd, tau_cmd[0], tau_cmd[1], tau_cmd[2]])
    omega_sq = M_inv @ wrench

    # Clamp omega² to be non-negative, then take sqrt
    omega_sq = np.clip(omega_sq, omega_min**2, omega_max**2)
    omega_cmd = np.sqrt(omega_sq)

    return omega_cmd
