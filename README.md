# Cooperative 4-UAV Cable-Suspended Load Simulation

A physically accurate, modular simulation of **4 quadrotor UAVs cooperatively
carrying a rectangular basket via 4 cables (one per corner)**, with both a
static point payload and a dynamic (sliding) mass on the basket floor.

---

## Features

### Physics Modelling
- **Full Newton–Euler rigid-body dynamics** for each drone and the basket load.
- **Geometric SO(3) attitude representation** using quaternions (no gimbal lock,
  no singularity) with rotation matrices for control laws (Lee, Leok &
  McClamroch, 2010).
- **Visco-elastic cable model** (spring-damper, unilateral: cables can only
  pull), with configurable stiffness tuned for <2% elongation at working
  tension. A rigid-constraint (Lagrange multiplier / Baumgarte stabilization)
  mode is also available.
- **Sliding mass dynamics** with full rotating-frame kinematics: Coriolis,
  centrifugal, and Euler pseudo-forces, plus smooth regularised Coulomb +
  viscous friction.
- **Off-CM cable attachment** on drones (creates a torque arm, coupling cable
  tension into attitude dynamics — physically important, not simplified away).
- **Parallel-axis theorem** for combining basket + static-load inertia at
  setup, with automatic CM shift and attachment point recomputation.

### Control Architecture
- **Mode A — Decentralized formation tracking**: each drone independently
  tracks a virtual point above its assigned corner, with cable-tension
  feedforward from static load sharing.
- **Mode B — Centralized load-wrench allocation**: a load-level SE(3)
  controller computes the desired wrench, which is distributed to 4 cable
  tensions via bounded least-squares (QP).
- Both modes use the same **geometric SE(3) controller** structure used in
  real flight controllers (PX4/ArduPilot implement linearised variants).

### Numerical Integration
- **Primary**: `scipy.integrate.solve_ivp` with `Radau` or `BDF` (implicit,
  A-stable, handles stiff cable dynamics).
- **Fallback**: Fixed-step RK4 for cross-validation and real-time loops.
- Quaternion renormalisation every step to prevent drift off SO(3).

### Visualization
- **3D Matplotlib animation**: drones (X-frame arms), cables (tension-colored),
  basket (wireframe box), sliding mass with trail.
- **9-panel telemetry plots**: tensions, positions, velocities, rotor speeds,
  energy, sliding mass trajectory.

---

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pyyaml

# Run default simulation (hover, formation mode, Radau integrator)
cd "Multi uav proto simulation"
python -m sim.run_sim

# Run with circular trajectory
python -m sim.run_sim --scenario circle --time 20

# Run with centralized control mode (edit config.yaml: control.mode: "centralized")
python -m sim.run_sim

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
Multi uav proto simulation/
├── dynamics/
│   ├── drone.py            # Single-UAV Newton-Euler + motor lag
│   ├── cable.py            # Elastic + rigid-constraint cable models
│   ├── load.py             # Basket rigid body + parallel-axis inertia
│   ├── sliding_mass.py     # Rotating-frame sliding mass dynamics
│   └── system.py           # Full state vector assembly, EOM, RK4 fallback
├── control/
│   ├── geometric_control.py  # SE(3) position + attitude controller
│   ├── allocation_qp.py      # Centralized tension allocation (bounded LS)
│   ├── formation_mode.py     # Mode A/B controllers + trajectory generators
│   └── mixer.py               # Rotor mixer matrix and inverse
├── sim/
│   ├── config.yaml           # All physical parameters (YAML)
│   └── run_sim.py            # Main entry point
├── viz/
│   ├── animate_3d.py         # 3D Matplotlib animation
│   └── plots.py              # Telemetry plotting
├── tests/
│   ├── test_single_cable_pendulum.py
│   ├── test_static_load_sharing.py
│   └── test_energy_sanity.py
└── README.md
```

---

## Configuration

All physical parameters are exposed in `sim/config.yaml`.  Key parameter
groups:

| Group      | Key Parameters                                      |
|------------|-----------------------------------------------------|
| `drone`    | mass, inertia, k_T, k_Q, motor lag, arm length      |
| `cable`    | natural length, stiffness, damping, model type       |
| `basket`   | mass, half-extents, static/dynamic load, friction    |
| `control`  | mode, PD gains, attitude gains, QP bounds            |
| `trajectory` | type, hover position, circle parameters            |
| `sim`      | integrator, tolerances, duration, output rate        |
| `wind`     | enable, mean velocity, gust amplitude/frequency      |

### Swapping in Real Hardware Parameters

To use real motor/propeller data:
1. Replace `k_T` and `k_Q` with values from thrust-stand testing of your
   actual motor/prop combo.
2. Replace `inertia_kgm2` with values from CAD or swing tests.
3. Replace `cable.stiffness_N_per_m` with measured stiffness of your rigging
   line (e.g. Dyneema/Spectra: ~70 kN/m for 2mm line).
4. Adjust `motor_time_constant_s` based on ESC response measurements.

---

## Stiff ODE / Cable Stiffness Tradeoff

Real synthetic cables (Dyneema, Kevlar) are nearly inextensible (stiffness
>50,000 N/m for typical rigging).  However, extremely high `k_cable` makes
the ODE very stiff, requiring either:
- A stiff-aware implicit integrator (Radau, BDF) — **our default approach**.
- Very small fixed-step sizes with explicit methods (RK4 at dt=1e-5 s).

The default `k_cable = 1500 N/m` gives ~0.9% elongation at hover tensions,
which is physically reasonable and numerically tractable.  For higher fidelity,
increase `k_cable` and ensure the integrator can handle it (Radau is robust
up to ~50,000 N/m with tight tolerances).

---

## What is Modelled vs. Simplified

### Modelled with Physical Justification
- Elastic spring-damper cable (with unilateral constraint: cables can't push)
- Newton–Euler rigid-body dynamics for drones and basket
- First-order motor lag (ESC + motor electrical dynamics)
- Off-CM cable attachment torque coupling on drones
- Rotating-frame sliding mass dynamics (Coriolis, centrifugal, Euler forces)
- Smooth Coulomb + viscous friction on sliding mass
- Parallel-axis inertia composition (basket + static load)
- Geometric SO(3) control (singularity-free, well-conditioned)
- Quadratic aerodynamic drag on drones

### Simplified for Tractability
- No blade-element rotor aerodynamics (lumped k_T, k_Q model)
- No ground-effect CFD (could be added as a height-dependent k_T modifier)
- No cable mass, drape, or sag (cable is massless, straight)
- No structural flexibility of drone frame or basket
- Simplified wind model (sinusoidal gust, not Dryden turbulence)
- No sensor noise or estimation errors (direct state feedback to controllers)

### Fidelity Margin Before Hardware Trials
The simulation captures the dominant dynamics of a cooperative slung-load
system.  For hardware deployment, the following should be addressed:
1. **Control loop discretization**: real autopilots (PX4) run at 250 Hz–1 kHz;
   the continuous-time controller here needs discretization (ZOH or Tustin).
2. **State estimation**: replace direct state feedback with estimator output
   (EKF/UKF with IMU + GPS/MOCAP).
3. **Communication latency**: inter-drone coordination has non-zero delay.
4. **Actuator-specific motor curves**: replace k_T/k_Q with full motor maps.

---

## Validation Tests

| Test | What it validates | Pass criterion |
|------|-------------------|----------------|
| `test_single_cable_pendulum` | Cable model pendulum period | < 2% error vs. analytic |
| `test_static_load_sharing` | Hover tension symmetry + CM shift | Tensions equal (symmetric), shift with off-center load |
| `test_energy_sanity` | No energy injection in passive drop | Energy never exceeds initial + tolerance |

---

## References

1. Lee, T., Leok, M., & McClamroch, N. H. (2010). "Geometric tracking control
   of a quadrotor UAV on SE(3)." IEEE CDC.
2. Goodarzi, F., Lee, D., & Lee, T. (2015). "Geometric control of a quadrotor
   UAV transporting a payload connected via flexible cable." Int. J. Control,
   Automation and Systems.
3. Sreenath, K., Michael, N., & Kumar, V. (2013). "Trajectory generation and
   control of a quadrotor with a cable-suspended load." IEEE ICRA.

---

## License

This project is provided as-is for research and educational purposes.
