"""
dynamics/cable.py — Cable force models (elastic spring-damper & rigid constraint).

Two cable models are provided, selectable via configuration:

1. **Elastic** (default): Visco-elastic unilateral spring-damper.
   Force is proportional to elongation beyond natural length, plus viscous damping.
   Cables can only pull (T ≥ 0); when slack (L < L0), T = 0.

2. **Rigid constraint**: Inextensible cable modelled as an index-3 DAE with
   Lagrange multipliers for tension and Baumgarte stabilization to prevent
   constraint drift.  Not integrated into the main ODE RHS directly; instead,
   the System assembler solves for tensions algebraically each step.

Both models return the tension magnitude and the unit direction vector from the
load attachment point toward the drone attachment point, so the calling code
can compute forces and torques on both bodies.

Stiffness/integrator tradeoff
-----------------------------
Real synthetic-fibre cables (Dyneema, Kevlar) are nearly inextensible.  We use
k_cable large enough that elongation stays under ~1-2 % of L0 at working
tensions, but not so large that the ODE becomes prohibitively stiff.  With
k_cable = 1500 N/m, nominal per-cable hover tension ~13.5 N gives δ ≈ 9 mm on
L0 = 1.2 m → 0.75 %.  The resulting stiff ODE is handled by Radau/BDF.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class CableParams:
    """Parameters for a single cable."""
    natural_length: float = 1.2          # L0 (m)
    stiffness: float = 1500.0            # k_cable (N/m)
    damping: float = 15.0                # c_cable (Ns/m)
    model: str = "elastic"               # "elastic" or "rigid_constraint"
    baumgarte_alpha: float = 10.0        # Baumgarte stabilization gain α
    baumgarte_beta: float = 10.0         # Baumgarte stabilization gain β


def compute_cable_force_elastic(
    p_drone_attach: np.ndarray,
    p_load_attach: np.ndarray,
    v_drone_attach: np.ndarray,
    v_load_attach: np.ndarray,
    params: CableParams,
) -> tuple[float, np.ndarray, float]:
    """Compute elastic cable tension and direction.

    Uses a smooth softplus function to model the unilateral constraint,
    preventing derivative discontinuities that bog down solvers.
    """
    e = p_drone_attach - p_load_attach          # vector from load to drone
    L = np.linalg.norm(e)

    if L < 1e-12:
        # Degenerate: attachment points coincident → no force
        return 0.0, np.array([0.0, 0.0, 1.0]), 0.0

    n_hat = e / L                                # unit direction load→drone

    # Smooth unilateral spring using softplus: delta = eps * ln(1 + exp((L - L0)/eps))
    eps_soft = 1.0e-4  # 0.1 mm transition width (sharp but differentiable)
    x = (L - params.natural_length) / eps_soft

    # Numerical guards against overflow
    if x > 50.0:
        delta_smooth = L - params.natural_length
        sigmoid = 1.0
    elif x < -50.0:
        delta_smooth = 0.0
        sigmoid = 0.0
    else:
        delta_smooth = eps_soft * np.log(1.0 + np.exp(x))
        sigmoid = 1.0 / (1.0 + np.exp(-x))

    # Project relative velocity to get rate of elongation
    v_rel = v_drone_attach - v_load_attach
    delta_dot = np.dot(v_rel, n_hat)

    # Tension is computed using smooth elongation and sigmoid-gated damping
    tension = params.stiffness * delta_smooth + params.damping * sigmoid * delta_dot

    # Final safety clamp (tension must be non-negative)
    tension = max(0.0, tension)

    return tension, n_hat, delta_smooth


def compute_cable_force_rigid(
    p_drone_attach: np.ndarray,
    p_load_attach: np.ndarray,
    v_drone_attach: np.ndarray,
    v_load_attach: np.ndarray,
    params: CableParams,
) -> tuple[float, np.ndarray, float]:
    """Compute constraint-level quantities for the rigid cable model.

    For the rigid model the tension is NOT computed here — it must be solved
    for via the constrained equations of motion (Lagrange multipliers).
    This function returns the constraint violation and stabilization terms
    that the system assembler needs.

    Returns
    -------
    constraint_violation : float — φ = 0.5 * (L² − L0²)
    n_hat               : (3,) — unit cable direction (load → drone)
    elongation           : float — (L − L0), for diagnostics
    """
    e = p_drone_attach - p_load_attach
    L = np.linalg.norm(e)

    if L < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0]), 0.0

    n_hat = e / L

    # Holonomic constraint: φ_i = 0.5 * (L² − L0²) = 0
    phi = 0.5 * (L**2 - params.natural_length**2)
    elongation = L - params.natural_length

    return phi, n_hat, elongation


def compute_rigid_constraint_terms(
    p_drone_attach: np.ndarray,
    p_load_attach: np.ndarray,
    v_drone_attach: np.ndarray,
    v_load_attach: np.ndarray,
    params: CableParams,
) -> dict:
    """Compute all Baumgarte-stabilized constraint terms for one cable.

    The constraint is φ = 0.5*(‖e‖² − L0²) = 0, where e = a_drone − r_load.

    Baumgarte stabilization replaces φ̈ = 0 with:
        φ̈ + 2α φ̇ + β² φ = 0

    This function returns φ, φ̇, the Jacobian contribution ∂φ/∂q, and the
    bias acceleration (everything except the unknown tension multiplier).
    The system assembler uses these to form and solve the linear system for
    the Lagrange multiplier (= cable tension).

    Returns
    -------
    dict with keys:
        'phi'       : constraint violation
        'phi_dot'   : constraint velocity violation
        'e'         : (3,) vector from load attach to drone attach
        'n_hat'     : (3,) unit cable direction
        'L'         : current cable length
        'baumgarte_rhs' : −2α φ̇ − β² φ  (scalar, right-hand side of stabilized constraint)
    """
    e = p_drone_attach - p_load_attach
    L = np.linalg.norm(e)

    if L < 1e-12:
        return {
            'phi': 0.0,
            'phi_dot': 0.0,
            'e': np.zeros(3),
            'n_hat': np.array([0.0, 0.0, 1.0]),
            'L': 0.0,
            'baumgarte_rhs': 0.0,
        }

    n_hat = e / L
    v_rel = v_drone_attach - v_load_attach

    phi = 0.5 * (L**2 - params.natural_length**2)
    phi_dot = np.dot(e, v_rel)               # = L * dL/dt

    alpha = params.baumgarte_alpha
    beta = params.baumgarte_beta
    baumgarte_rhs = -2.0 * alpha * phi_dot - beta**2 * phi

    return {
        'phi': phi,
        'phi_dot': phi_dot,
        'e': e,
        'n_hat': n_hat,
        'L': L,
        'baumgarte_rhs': baumgarte_rhs,
    }


def cable_force_on_load(tension: float, n_hat: np.ndarray) -> np.ndarray:
    """Force exerted on the load by cable i: F = T * n̂  (pulls load toward drone)."""
    return tension * n_hat


def cable_force_on_drone(tension: float, n_hat: np.ndarray) -> np.ndarray:
    """Reaction force on drone from cable i: F = -T * n̂."""
    return -tension * n_hat
