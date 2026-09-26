# %% Imports and experiment settings
import argparse
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import cvxpy as cp
import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.signal import iirdesign, sosfiltfilt
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    import meshcat

Array = NDArray[np.float64]
IndexArray = NDArray[np.intp]
DOF = 3
NPARAM = 10
ENCODER_RATE_HZ = 100
PRIME_ANGULAR_FREQUENCIES = np.array([2.0, 3.0, 5.0])  # rad/s; one per joint
JOINT_AMPLITUDES = np.array([0.65, 0.6, 0.5])
FILTER_CUTOFF_MULTIPLIER = 8.0
FILTER_STEEPNESS = 0.85
FILTER_PASS_LOSS_DB = 0.1
FILTER_STOP_ATTENUATION_DB = 60.0
FILTER_MARGIN_S = 0.5
ENCODER_NOISE_STD_RAD = 0.001
TORQUE_NOISE_FRACTION = 0.03
DURATION_S = 45
FULL_SAMPLES = 1000
DOWN_SAMPLES = 100


@dataclass(frozen=True)
class RobotSettings:
    """Geometry and known dynamics of the three-joint experiment arm."""

    yaw_position: tuple[float, float, float] = (0.0, 0.0, 0.2)
    yaw_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pitch_position: tuple[float, float, float] = (0.43, 0.12, 0.17)
    pitch_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    terminal_position: tuple[float, float, float] = (0.34, 0.09, 0.16)
    roll_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    terminal_mass: float = 1.15
    terminal_com: tuple[float, float, float] = (0.11, 0.045, -0.065)
    terminal_central_inertia: tuple[tuple[float, float, float], ...] = (
        (0.055, 0.004, -0.003),
        (0.004, 0.062, 0.005),
        (-0.003, 0.005, 0.074),
    )


ROBOT = RobotSettings()


# %% Parameter convention and physical consistency
@dataclass(frozen=True)
class Samples:
    """One sample per leading axis; regressors are affine in terminal inertia."""

    q: Array
    tau: Array
    reg: Array
    offset: Array
    mass_reg: Array
    mass_offset: Array

    def select(self, indices: IndexArray) -> "Samples":
        """Take the same subset from every measurement and regressor."""
        return Samples(
            self.q[indices],
            self.tau[indices],
            self.reg[indices],
            self.offset[indices],
            self.mass_reg[indices],
            self.mass_offset[indices],
        )


def parameters(mass: float, com: Array, central: Array) -> Array:
    """Return [m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz] at the roll joint."""
    inertia = central + mass * ((com @ com) * np.eye(3) - np.outer(com, com))
    return np.array(
        [
            mass,
            *(mass * com),
            inertia[0, 0],
            inertia[0, 1],
            inertia[1, 1],
            inertia[0, 2],
            inertia[1, 2],
            inertia[2, 2],
        ]
    )


def pseudo_inertia(theta: Array) -> Array:
    """Return the 4x4 pseudo inertia used for physical consistency.

    This matrix is affine in theta; positive semidefiniteness enforces a
    physically consistent inertia.
    """
    m, hx, hy, hz, ixx, ixy, iyy, ixz, iyz, izz = theta
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return np.block(
        [
            [np.trace(inertia) / 2 * np.eye(3) - inertia, np.array([[hx], [hy], [hz]])],
            [np.array([[hx, hy, hz]]), np.array([[m]])],
        ]
    )


PSEUDO_BASIS = np.stack([pseudo_inertia(basis) for basis in np.eye(NPARAM)], axis=-1)


# %% Synthetic arm and MuJoCo inverse dynamics
def build_arm(theta: Array, robot: RobotSettings = ROBOT) -> mujoco.MjModel:
    """Build a three-joint MuJoCo arm with massless links."""
    mass = theta[0]
    if mass <= 0:
        raise ValueError("Terminal body requires positive mass")
    com = theta[1:4] / mass
    ixx, ixy, iyy, ixz, iyz, izz = theta[4:]
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    central = inertia - mass * ((com @ com) * np.eye(3) - np.outer(com, com))
    if np.linalg.eigvalsh(central).min() <= 0:
        raise ValueError("Terminal body requires positive mass and central inertia")
    def vec(values: tuple[float, float, float] | Array) -> str:
        return " ".join(f"{v:.17g}" for v in values)

    pos = vec(com)
    full = " ".join(
        f"{v:.17g}"
        for v in (
            central[0, 0],
            central[1, 1],
            central[2, 2],
            central[0, 1],
            central[0, 2],
            central[1, 2],
        )
    )
    xml = f"""
    <mujoco model="offset_yaw_pitch_roll">
      <option gravity="{vec(robot.gravity)}"/>
      <worldbody>
        <body name="yaw_link" pos="{vec(robot.yaw_position)}">
          <joint name="yaw" type="hinge" axis="{vec(robot.yaw_axis)}"/>
          <inertial pos="0 0 0" mass="1e-8" diaginertia="1e-8 1e-8 1e-8"/>
          <body name="pitch_link" pos="{vec(robot.pitch_position)}">
            <joint name="pitch" type="hinge" axis="{vec(robot.pitch_axis)}"/>
            <inertial pos="0 0 0" mass="1e-8" diaginertia="1e-8 1e-8 1e-8"/>
            <body name="end_effector" pos="{vec(robot.terminal_position)}">
              <joint name="roll" type="hinge" axis="{vec(robot.roll_axis)}"/>
              <inertial pos="{pos}" mass="{mass:.17g}" fullinertia="{full}"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    # MuJoCo requires positive inertia while compiling joint bodies. Remove
    # these temporary values and refresh constants: the links are massless.
    model.body_mass[1:3] = 0
    model.body_inertia[1:3] = 0
    mujoco.mj_setConst(model, mujoco.MjData(model))
    assert model.nq == model.nv == DOF
    return model


def response(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: Array,
    velocity: Array,
    acceleration: Array,
) -> tuple[Array, Array]:
    data.qpos[:] = q
    data.qvel[:] = velocity
    data.qacc[:] = acceleration
    mujoco.mj_inverse(model, data)
    mass = np.empty((DOF, DOF))
    mujoco.mj_fullM(model, data, mass)
    return data.qfrc_inverse.copy(), mass


# %% Affine inverse-dynamics and mass-matrix regressors
def skew(vector: Array) -> Array:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def spatial_inertia_bases() -> Array:
    """Exact derivative of body spatial inertia with respect to its ten parameters."""
    bases = np.zeros((NPARAM, 6, 6))
    bases[0, :3, :3] = np.eye(3)
    for axis in range(3):
        first_moment = np.eye(3)[axis]
        bases[axis + 1, :3, 3:] = -skew(first_moment)
        bases[axis + 1, 3:, :3] = skew(first_moment)
    for index, (row, column) in enumerate(
        ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)), start=4
    ):
        bases[index, row + 3, column + 3] = 1.0
        bases[index, column + 3, row + 3] = 1.0
    return bases


SPATIAL_BASES = spatial_inertia_bases()


def analytic_body_regressors(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> tuple[Array, Array]:
    """Return torque and mass-matrix regressors for one body origin.

    For each inertial basis, project spatial Newton-Euler force through the
    body Jacobian. The mass basis is J.T @ spatial_inertia_basis @ J.
    Call after ``mj_inverse`` and ``mj_rnePostConstraint``.
    """
    velocity = np.zeros(6)
    acceleration = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 1
    )
    mujoco.mj_objectAcceleration(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, acceleration, 1
    )
    jacobian_linear = np.zeros((3, DOF))
    jacobian_angular = np.zeros((3, DOF))
    mujoco.mj_jacBody(model, data, jacobian_linear, jacobian_angular, body_id)
    rotation = data.xmat[body_id].reshape(3, 3)
    jacobian = np.vstack((rotation.T @ jacobian_linear, rotation.T @ jacobian_angular))

    linear_velocity, angular_velocity = velocity[3:], velocity[:3]
    spatial_velocity = np.r_[linear_velocity, angular_velocity]
    spatial_acceleration = np.r_[
        acceleration[3:] - np.cross(angular_velocity, linear_velocity), acceleration[:3]
    ]
    force_cross = np.block(
        [
            [skew(angular_velocity), np.zeros((3, 3))],
            [skew(linear_velocity), skew(angular_velocity)],
        ]
    )
    momenta = np.einsum("kij,j->ki", SPATIAL_BASES, spatial_velocity)
    forces = np.einsum("kij,j->ki", SPATIAL_BASES, spatial_acceleration)
    forces += np.einsum("ij,kj->ki", force_cross, momenta)
    torque_regressor = jacobian.T @ forces.T
    mass_regressor = np.einsum("ai,kab,bj->ijk", jacobian, SPATIAL_BASES, jacobian)
    return torque_regressor, mass_regressor


# %% Excitation, filtering, and measurements
def trajectory(start: float, stop: float, count: int) -> tuple[Array, Array, Array]:
    """One sinusoid per joint, at prime angular frequencies."""
    t = np.linspace(start, stop, count, endpoint=False)[:, None]
    omega = PRIME_ANGULAR_FREQUENCIES
    q = JOINT_AMPLITUDES * np.sin(omega * t)
    v = JOINT_AMPLITUDES * omega * np.cos(omega * t)
    a = -JOINT_AMPLITUDES * omega**2 * np.sin(omega * t)
    return q, v, a


def signal_filter(sample_rate_hz: int) -> Array:
    """Design an elliptic low-pass filter for noisy measurements."""
    cutoff = FILTER_CUTOFF_MULTIPLIER * np.max(PRIME_ANGULAR_FREQUENCIES) / (2 * np.pi)
    return iirdesign(
        wp=cutoff,
        ws=cutoff + (0.99 - 0.98 * FILTER_STEEPNESS) * (sample_rate_hz / 2 - cutoff),
        gpass=FILTER_PASS_LOSS_DB,
        gstop=FILTER_STOP_ATTENUATION_DB,
        ftype="ellip",
        output="sos",
        fs=sample_rate_hz,
    )


def filter_encoder(positions: Array, sample_rate_hz: int) -> tuple[Array, Array, Array]:
    """Differentiate encoder readings, then filter position and derivatives."""
    seconds = 1 / sample_rate_hz
    raw_velocity = np.gradient(positions, seconds, axis=0)
    raw_acceleration = np.gradient(raw_velocity, seconds, axis=0)
    sos = signal_filter(sample_rate_hz)
    return (
        sosfiltfilt(sos, positions, axis=0),
        sosfiltfilt(sos, raw_velocity, axis=0),
        sosfiltfilt(sos, raw_acceleration, axis=0),
    )


def generate_samples(
    start: float,
    stop: float,
    count: int,
    torque_noise_fraction: float,
    rng: np.random.Generator,
    truth: Array,
    encoder_noise: float = 0.0,
    robot: RobotSettings = ROBOT,
) -> Samples:
    """Filter dense noisy signals, then select training or test samples.

    The torque noise standard deviation is a fraction of each joint's peak
    absolute torque. Regressors use filtered encoder states; measured torques
    remain separate so the same noisy data can be fit by every method.
    """
    dense_count = round((stop - start) * ENCODER_RATE_HZ)
    margin = round(FILTER_MARGIN_S * ENCODER_RATE_HZ)
    if dense_count < ENCODER_RATE_HZ * 2 or count > dense_count - 2 * margin:
        raise ValueError("Trajectory is too short for filtering and downsampling")
    true_q, true_v, true_a = trajectory(start, stop, dense_count)
    actual = build_arm(truth, robot)
    actual_data = mujoco.MjData(actual)
    true_tau = np.stack(
        [
            response(actual, actual_data, q, v, a)[0]
            for q, v, a in zip(true_q, true_v, true_a, strict=True)
        ]
    )
    if encoder_noise or torque_noise_fraction:
        noisy_q = true_q + encoder_noise * rng.standard_normal(true_q.shape)
        torque_noise_std = torque_noise_fraction * np.max(np.abs(true_tau), axis=0)
        noisy_tau = true_tau + torque_noise_std * rng.standard_normal(true_tau.shape)
        measured_q, measured_v, measured_a = filter_encoder(noisy_q, ENCODER_RATE_HZ)
        measured_tau = sosfiltfilt(signal_filter(ENCODER_RATE_HZ), noisy_tau, axis=0)
    else:
        measured_q, measured_v, measured_a = true_q, true_v, true_a
        measured_tau = true_tau
    indices = np.linspace(margin, dense_count - margin - 1, count, dtype=int)
    q_all, v_all, a_all = measured_q[indices], measured_v[indices], measured_a[indices]
    measured_tau = measured_tau[indices]
    terminal_id = mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    tau = np.empty((count, DOF))
    reg = np.empty((count, DOF, NPARAM))
    offset = np.zeros((count, DOF))
    mass_reg = np.empty((count, DOF, DOF, NPARAM))
    mass_offset = np.zeros((count, DOF, DOF))
    for s, (q, v, a) in enumerate(zip(q_all, v_all, a_all, strict=True)):
        measured_state_tau, measured_state_mass = response(actual, actual_data, q, v, a)
        mujoco.mj_rnePostConstraint(actual, actual_data)
        reg[s], mass_reg[s] = analytic_body_regressors(actual, actual_data, terminal_id)
        # At the filtered encoder state, compare each affine model with MuJoCo.
        # Sensor torque is noisy, so it is not used in this identity check.
        np.testing.assert_allclose(
            offset[s] + reg[s] @ truth, measured_state_tau, rtol=2e-6, atol=2e-8
        )
        np.testing.assert_allclose(
            mass_offset[s] + np.einsum("ijk,k->ij", mass_reg[s], truth),
            measured_state_mass,
            rtol=2e-6,
            atol=2e-8,
        )
        tau[s] = measured_tau[s]
    mass_reg = (mass_reg + mass_reg.swapaxes(1, 2)) / 2
    mass_offset = (mass_offset + mass_offset.swapaxes(1, 2)) / 2
    return Samples(q_all, tau, reg, offset, mass_reg, mass_offset)


# %% Three convex identification objectives
def observable_rank(reg: Array) -> tuple[int, float]:
    """Normalize columns before determining the dimensionless rank."""
    flat = reg.reshape(-1, NPARAM)
    norms = np.linalg.norm(flat, axis=0)
    if np.any(norms == 0):
        return 0, 0.0
    singular = np.linalg.svd(flat / norms, compute_uv=False)
    ratio = float(singular[-1] / singular[0])
    return int(np.sum(singular > singular[0] * 1e-6)), ratio


def residual_whitener(samples: Samples, ls: Array) -> Array:
    """Estimate one constant torque whitening matrix from LS residuals."""
    count = len(samples.q)
    ls_error = samples.offset + np.einsum("sij,j->si", samples.reg, ls) - samples.tau
    covariance = ls_error.T @ ls_error / count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(1e-8, float(eigenvalues.max()) * 1e-4)
    return (eigenvectors / np.sqrt(np.maximum(eigenvalues, floor))) @ eigenvectors.T


def fit_fusion_dual(samples: Samples) -> Array:
    """Solve the dual-metric semidefinite program in MOSEK's dual form."""
    import mosek.fusion as mf
    import mosek.fusion.pythonic  # noqa: F401  (enables Fusion slicing)

    count = len(samples.q)
    flat_reg = samples.reg.reshape(-1, NPARAM)
    target = (samples.tau - samples.offset).reshape(-1)

    # Solve the dual of the per-sample Schur formulation. Each PSD variable
    # corresponds to one sample; the equality dual yields the ten parameters.
    with mf.Model("dual_metric_dual") as model:
        sample_dual = model.variable("sample_dual", mf.Domain.inPSDCone(DOF + 1, count))
        matrix_dual = sample_dual[:, :DOF, :DOF].reshape([count * DOF * DOF])
        residual_dual = sample_dual[:, :DOF, DOF].reshape([count * DOF])
        slack_dual = sample_dual[:, DOF, DOF].reshape([count])
        model.constraint(slack_dual, mf.Domain.equalsTo(1 / count))

        inertia_dual = model.variable("inertia_dual", mf.Domain.inPSDCone(4))
        terms = []
        for k in range(NPARAM):
            terms.append(
                mf.Expr.add(
                    [
                        mf.Expr.dot(matrix_dual, samples.mass_reg[..., k].reshape(-1)),
                        mf.Expr.mul(2, mf.Expr.dot(residual_dual, flat_reg[:, k])),
                        mf.Expr.dot(
                            inertia_dual, mf.Matrix.dense(PSEUDO_BASIS[..., k])
                        ),
                    ]
                )
            )
        stationarity = model.constraint(
            "stationarity", mf.Expr.vstack(terms), mf.Domain.equalsTo(0.0)
        )
        objective = mf.Expr.sub(
            mf.Expr.dot(matrix_dual, samples.mass_offset.reshape(-1)),
            mf.Expr.mul(2, mf.Expr.dot(residual_dual, target)),
        )
        model.objective(mf.ObjectiveSense.Minimize, objective)
        model.solve()
        if model.getPrimalSolutionStatus() != mf.SolutionStatus.Optimal:
            raise RuntimeError(f"dual_metric_dual: {model.getPrimalSolutionStatus()}")
        return -np.asarray(stationarity.dual()).reshape(-1)


def fit_fusion_primal(samples: Samples) -> Array:
    """Reference Schur formulation; Fusion's dual form solves faster at scale."""
    import mosek.fusion as mf

    count = len(samples.q)
    with mf.Model("dual_metric_primal") as model:
        x = model.variable("coefficients", NPARAM, mf.Domain.unbounded())
        pseudo = mf.Expr.reshape(
            mf.Expr.mul(PSEUDO_BASIS.reshape(16, NPARAM), x), [4, 4]
        )
        model.constraint("physical_inertia", pseudo, mf.Domain.inPSDCone(4))
        residual = mf.Expr.sub(
            mf.Expr.mul(samples.reg.reshape(-1, NPARAM), x),
            (samples.tau - samples.offset).reshape(-1),
        )
        mass = mf.Expr.reshape(
            mf.Expr.add(
                mf.Expr.mul(samples.mass_reg.reshape(-1, NPARAM), x),
                samples.mass_offset.reshape(-1),
            ),
            [count, DOF, DOF],
        )
        column = mf.Expr.reshape(residual, [count, DOF, 1])
        row = mf.Expr.reshape(residual, [count, 1, DOF])
        slacks = model.variable("sample_cost", count, mf.Domain.greaterThan(0.0))
        # [[M(q, x), r(x)], [r(x).T, s]] >= 0 means s >= r.T M^-1 r.
        block = mf.Expr.stack(
            1,
            [
                mf.Expr.stack(2, [mass, column]),
                mf.Expr.stack(2, [row, mf.Expr.reshape(slacks, [count, 1, 1])]),
            ],
        )
        model.constraint("sample_bound", block, mf.Domain.inPSDCone(DOF + 1, count))
        model.objective(mf.ObjectiveSense.Minimize, mf.Expr.sum(slacks))
        model.solve()
        if model.getPrimalSolutionStatus() != mf.SolutionStatus.Optimal:
            raise RuntimeError(f"dual_metric_primal: {model.getPrimalSolutionStatus()}")
        return np.asarray(x.level()).copy()


def cvx_physical(x: cp.Variable) -> list[cp.Constraint]:
    pseudo = cp.reshape(PSEUDO_BASIS.reshape(16, NPARAM) @ x, (4, 4), order="C")
    return [pseudo >> 0]


def solve_cvxpy(
    cost: cp.Expression, x: cp.Variable, constraints: list[cp.Constraint]
) -> Array:
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver="CLARABEL", verbose=False)
    if problem.status != cp.OPTIMAL or x.value is None:
        raise RuntimeError(f"Identification failed: {problem.status}")
    return np.asarray(x.value).copy()


def fit_baselines(samples: Samples) -> dict[str, Array]:
    """Fit constrained least squares and covariance-weighted least squares."""
    count = len(samples.q)
    x = cp.Variable(NPARAM)
    physical = cvx_physical(x)
    flat_reg = samples.reg.reshape(-1, NPARAM)
    target = (samples.tau - samples.offset).reshape(-1)
    residual = flat_reg @ x - target
    ls = solve_cvxpy(cp.sum_squares(residual), x, physical)
    whiten = residual_whitener(samples, ls)
    weighted = cp.reshape(residual, (count, DOF), order="C") @ whiten.T
    wls = solve_cvxpy(cp.sum_squares(weighted), x, physical)
    return {"LS": ls, "WLS": wls}


def fit_cvxpy_dual(samples: Samples) -> Array:
    """Solve the per-sample Schur constraints with CVXPY and Clarabel."""
    count = len(samples.q)
    x = cp.Variable(NPARAM)
    physical = cvx_physical(x)

    # [[M, r], [r.T, s]] >= 0 gives s >= r.T @ inv(M) @ r.
    slacks = cp.Variable(count, nonneg=True)
    blocks: list[cp.Constraint] = []
    for s in range(count):
        mass = samples.mass_offset[s] + cp.reshape(
            samples.mass_reg[s].reshape(DOF * DOF, NPARAM) @ x, (DOF, DOF), order="C"
        )
        r = samples.reg[s] @ x + samples.offset[s] - samples.tau[s]
        column = cp.reshape(r, (DOF, 1), order="C")
        block = cp.bmat(
            [
                [mass, column],
                [column.T, cp.reshape(slacks[s], (1, 1), order="C")],
            ]
        )
        blocks.append(block >> 0)
    return solve_cvxpy(cp.sum(slacks) / count, x, physical + blocks)


def fit(samples: Samples, solver: str = "auto") -> dict[str, Array]:
    """Fit baselines in CVXPY; choose a backend for the dual-metric fit.

    ``MOSEK`` selects the faster Fusion dual form. ``MOSEK_PRIMAL`` runs its
    reference primal form. ``auto`` falls back to Clarabel if MOSEK is absent.
    """
    if solver not in {"auto", "CLARABEL", "MOSEK", "MOSEK_PRIMAL"}:
        raise ValueError(f"Unknown solver: {solver}")
    estimates = fit_baselines(samples)
    if solver == "CLARABEL":
        estimates["Dual metric"] = fit_cvxpy_dual(samples)
        return estimates
    try:
        import mosek
        import mosek.fusion as mf
    except ImportError:
        if solver in {"MOSEK", "MOSEK_PRIMAL"}:
            raise
        estimates["Dual metric"] = fit_cvxpy_dual(samples)
        return estimates
    try:
        estimates["Dual metric"] = (
            fit_fusion_primal(samples)
            if solver == "MOSEK_PRIMAL"
            else fit_fusion_dual(samples)
        )
    except (mosek.Error, mf.OptimizeError, mf.SolutionError):
        if solver in {"MOSEK", "MOSEK_PRIMAL"}:
            raise
        estimates["Dual metric"] = fit_cvxpy_dual(samples)
    return estimates


# %% Evaluation and one reproducible comparison
def geodesic_distance(reference: Array, estimated: Array) -> float:
    """Return the pseudo-inertia geodesic length (half-trace convention)."""
    ref = pseudo_inertia(reference)
    est = pseudo_inertia(estimated)
    values, vectors = np.linalg.eigh(ref)
    if values.min() <= 0:
        raise ValueError("Reference pseudo inertia must be positive definite")
    if np.linalg.eigvalsh(est).min() <= 0:
        return float("inf")
    inv_sqrt = (vectors / np.sqrt(values)) @ vectors.T
    relative = inv_sqrt @ est @ inv_sqrt
    logs = np.log(np.linalg.eigvalsh(relative))
    return float(np.sqrt(0.5 * logs @ logs))


def true_parameters(robot: RobotSettings = ROBOT) -> Array:
    """Return the terminal-body inertia used to generate synthetic data."""
    return parameters(
        robot.terminal_mass,
        np.asarray(robot.terminal_com),
        np.asarray(robot.terminal_central_inertia),
    )


# %% Optional interactive robot view (install meshcat in the notebook)
def show_robot(
    q: Array | None = None,
    viewer: "meshcat.Visualizer | None" = None,
    *,
    robot: RobotSettings = ROBOT,
) -> "meshcat.Visualizer":
    """Show the arm in Meshcat; pass the returned viewer to update its pose."""
    import meshcat
    from meshcat import geometry

    if q is None:
        q = np.array([0.35, -0.4, 0.25])
    q = np.asarray(q, dtype=float)
    if q.shape != (DOF,):
        raise ValueError(f"Expected {DOF} joint angles")
    model = build_arm(true_parameters(robot), robot)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("yaw_link", "pitch_link", "end_effector")
    ]
    points = [data.xpos[body_id].copy() for body_id in body_ids]
    points.append(data.xipos[body_ids[-1]].copy())  # End-effector COM.

    new_viewer = viewer is None
    if viewer is None:
        viewer = meshcat.Visualizer()
    joint_material = geometry.MeshLambertMaterial(color=0x3268A8)
    link_material = geometry.MeshLambertMaterial(color=0x68A7C2)
    for name, point in zip(("yaw", "pitch", "roll", "com"), points, strict=True):
        node = viewer["robot"][name]
        material = (
            geometry.MeshLambertMaterial(color=0xE67736)
            if name == "com"
            else joint_material
        )
        node.set_object(geometry.Sphere(0.055 if name == "com" else 0.035), material)
        transform = np.eye(4)
        transform[:3, 3] = point
        node.set_transform(transform)
    for index, (start, end) in enumerate(pairwise(points)):
        direction = end - start
        length = np.linalg.norm(direction)
        rotation, _ = Rotation.align_vectors([direction / length], [[0, 1, 0]])
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = (start + end) / 2
        node = viewer["robot"][f"link_{index}"]
        node.set_object(geometry.Cylinder(length, 0.022), link_material)
        node.set_transform(transform)

    if new_viewer:
        try:
            import google.colab  # noqa: F401  (detect the Colab runtime)
        except ImportError:
            try:
                from IPython import get_ipython
                from IPython.display import display
            except ImportError:
                pass
            else:
                if get_ipython() is not None:
                    display(viewer.jupyter_cell(height=480))
        else:
            from IPython.display import display

            # Colab's port proxy does not reliably forward MeshCat's WebSocket.
            # Bundle the current scene into the iframe instead.
            display(viewer.render_static(height=480))
    return viewer


def main(
    *, seed: int = 42, solver: str = "auto", robot: RobotSettings = ROBOT
) -> dict[str, dict[str, Array]]:
    """Fit one noisy trajectory at full and downsampled resolution."""
    truth = true_parameters(robot)
    full = generate_samples(
        0,
        DURATION_S,
        FULL_SAMPLES,
        TORQUE_NOISE_FRACTION,
        np.random.default_rng(seed),
        truth,
        encoder_noise=ENCODER_NOISE_STD_RAD,
        robot=robot,
    )
    downsampled = full.select(np.linspace(0, len(full.q) - 1, DOWN_SAMPLES, dtype=int))
    results = {}
    print(f"Seed {seed}; massless links; end-effector inertia; solver {solver}")
    print(f"{'Dataset':<13} {'Samples':>7} {'Method':<13} {'Geodesic distance':>18}")
    for label, samples in (("Full", full), ("Downsampled", downsampled)):
        rank, _ = observable_rank(samples.reg)
        if rank != NPARAM:
            raise RuntimeError(f"{label} regressor rank is {rank}/{NPARAM}")
        results[label] = fit(samples, solver=solver)
        for method, estimate in results[label].items():
            print(
                f"{label:<13} {len(samples.q):>7} {method:<13} "
                f"{geodesic_distance(truth, estimate):18.6f}"
            )
    return results


# %% Command-line entry point (notebook readers call main() above)
if __name__ == "__main__" and "__file__" in globals():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=("auto", "CLARABEL", "MOSEK", "MOSEK_PRIMAL"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show-robot", action="store_true", help="Open MeshCat after fitting"
    )
    args = parser.parse_args()
    main(seed=args.seed, solver=args.solver)
    if args.show_robot:
        viewer = show_robot().open()
        print(f"MeshCat robot view: {viewer.url()} (Ctrl+C to close)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
