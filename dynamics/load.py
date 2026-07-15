"""
dynamics/load.py — Basket (load) rigid-body setup and dynamics helpers.

The basket is a rigid rectangular body with:
  • An empty-basket mass and inertia.
  • A *static* point load bolted rigidly to the basket, folded into the combined
    mass/inertia at setup via the parallel-axis theorem.
  • Four corner cable attachment points at body-frame positions q_i.

All quantities are expressed relative to the *combined* center of mass (basket +
static load) after the CM-shift recomputation.

Coordinate convention: body frame L, origin at combined CM, axes aligned with
basket edges (x along length, y along width, z up).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BasketParams:
    """Combined load parameters (basket + static payload)."""

    # --- raw configuration inputs ---
    empty_mass: float = 1.2               # basket empty mass (kg)
    half_extent: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.3])
    )                                      # [a/2, b/2] (m)
    corner_height: float = 0.15           # height of corner attach above CM (m)

    static_load_mass: float = 1.5         # kg
    static_load_pos_body: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.05])
    )                                      # body-frame position BEFORE CM shift

    dynamic_load_mass: float = 0.8        # kg
    floor_height: float = -0.10           # floor height in body frame (m)
    mu_k: float = 0.3                     # kinetic friction coefficient
    mu_s: float = 0.4                     # static friction coefficient
    visc_damping: float = 0.5             # viscous damping (Ns/m)
    friction_v_scale: float = 0.01        # regularisation velocity (m/s)

    # --- computed at setup ---
    total_mass: float = field(init=False)
    J_combined: np.ndarray = field(init=False)
    J_combined_inv: np.ndarray = field(init=False)
    corner_points_body: np.ndarray = field(init=False)  # (4, 3) in combined CM frame
    static_load_pos_combined: np.ndarray = field(init=False)
    cm_shift: np.ndarray = field(init=False)

    def __post_init__(self):
        self.half_extent = np.asarray(self.half_extent, dtype=float)
        self.static_load_pos_body = np.asarray(self.static_load_pos_body, dtype=float)
        self._compute_combined_properties()

    def _compute_combined_properties(self):
        """Recompute combined CM, inertia, and attachment points."""
        m_b = self.empty_mass
        m_s = self.static_load_mass
        self.total_mass = m_b + m_s

        # ---- CM shift ----
        # Basket CM is at body-frame origin (by definition before shift).
        # Static load is at static_load_pos_body.
        # Combined CM in the old frame:
        self.cm_shift = (m_s / self.total_mass) * self.static_load_pos_body

        # ---- Rewrite static load position relative to new CM ----
        self.static_load_pos_combined = self.static_load_pos_body - self.cm_shift

        # ---- Basket inertia about its own CM (thin rectangular plate approx) ----
        a, b = 2 * self.half_extent[0], 2 * self.half_extent[1]
        # Assuming small thickness h ≈ 0.05 m for the plate
        h = 0.05
        J_basket_cm = np.diag([
            m_b / 12 * (b**2 + h**2),
            m_b / 12 * (a**2 + h**2),
            m_b / 12 * (a**2 + b**2),
        ])

        # ---- Shift basket inertia to combined CM (parallel axis theorem) ----
        d_basket = -self.cm_shift   # basket old CM relative to new combined CM
        J_basket_combined = _parallel_axis(J_basket_cm, m_b, d_basket)

        # ---- Static load inertia about combined CM (point mass) ----
        d_static = self.static_load_pos_combined
        J_static_combined = _parallel_axis(np.zeros((3, 3)), m_s, d_static)

        # ---- Combined inertia ----
        self.J_combined = J_basket_combined + J_static_combined
        self.J_combined_inv = np.linalg.inv(self.J_combined)

        # ---- Corner attachment points in combined CM frame ----
        ax, ay = self.half_extent
        hz = self.corner_height
        # Corners: (+a/2, +b/2), (-a/2, +b/2), (-a/2, -b/2), (+a/2, -b/2)
        self.corner_points_body = np.array([
            [ ax,  ay, hz],
            [-ax,  ay, hz],
            [-ax, -ay, hz],
            [ ax, -ay, hz],
        ]) - self.cm_shift  # shift to combined CM


def _parallel_axis(J_cm: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
    """Parallel axis theorem: J_new = J_cm + m * (d·d I₃ − d dᵀ)."""
    return J_cm + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


# ---------------------------------------------------------------------------
# Load state packing / unpacking
# ---------------------------------------------------------------------------

LOAD_STATE_DIM = 13  # 3 pos + 3 vel + 4 quat + 3 omega


def pack_load_state(p: np.ndarray, v: np.ndarray, q: np.ndarray,
                    omega: np.ndarray) -> np.ndarray:
    """Pack load state into flat array (13 elements)."""
    return np.concatenate([p, v, q, omega])


def unpack_load_state(x: np.ndarray):
    """Unpack flat array into (p, v, q, omega)."""
    return x[0:3], x[3:6], x[6:10], x[10:13]


# ---------------------------------------------------------------------------
# Load corner positions / velocities in world frame
# ---------------------------------------------------------------------------

def get_corner_world(p_L: np.ndarray, R_L: np.ndarray,
                     corner_body: np.ndarray) -> np.ndarray:
    """World-frame position of a basket corner.

    r_i = p_L + R_L · q_i
    """
    return p_L + R_L @ corner_body


def get_corner_velocity_world(
    v_L: np.ndarray, omega_L_body: np.ndarray,
    R_L: np.ndarray, corner_body: np.ndarray,
) -> np.ndarray:
    """World-frame velocity of a basket corner.

    ṙ_i = v_L + (R_L ω_L) × (R_L q_i)
    """
    omega_world = R_L @ omega_L_body
    r_offset_world = R_L @ corner_body
    return v_L + np.cross(omega_world, r_offset_world)
