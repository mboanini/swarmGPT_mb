"""Backend module for the swarm_gpt web app."""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import numpy as np
import yaml
from scipy.interpolate import make_smoothing_spline

import swarm_gpt.core.motion_primitives as _mp_module

from swarm_gpt.core import Choreographer, DroneController
from swarm_gpt.core.agent_b.pipeline import Pipeline
from swarm_gpt.core.agent_b.primitive_writer import register_with_router, write_descriptions_to_yaml, write_to_file
from swarm_gpt.core.sim import simulate_axswarm
from swarm_gpt.exception import LLMException

_PRIMITIVES_PATH = Path(__file__).resolve().parent / "motion_primitives.py"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

colors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.7, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.5],
]

P = ParamSpec("P")
R = TypeVar("R")


def correct_velocity(positions, nominal_velocity, dt):
    """Filter one swarm (N, 3), checking OSQP's result in the input precision."""
    from osqp import OSQPException
    from qpsolvers import solve_qp
    from qpsolvers.exceptions import SolverError
    from scipy import sparse

    positions = np.asarray(positions)
    nominal_velocity = np.asarray(nominal_velocity)
    dtype = np.result_type(positions.dtype, nominal_velocity.dtype, np.float32)
    positions = positions.astype(dtype, copy=False)
    nominal_velocity = nominal_velocity.astype(dtype, copy=False)
    n = len(positions)
    d_safe, kz, alpha, tol = 0.20, 2.5, 4.0, 1e-4  # Same as CBFSim.
    w = np.array([1.0, 1.0, 1.0 / kz**2], dtype=dtype)
    G, hb = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            h_val = np.sum(w * d**2) - dtype.type(d_safe**2)
            Lf_h = 0
            g_i = np.zeros((3, 3 * n), dtype=dtype)
            g_j = np.zeros((3, 3 * n), dtype=dtype)
            g_i[:, 3*i:3*i+3] = np.eye(3)
            g_j[:, 3*j:3*j+3] = np.eye(3)
            Lg_h = (2.0 * w * d) @ (g_i - g_j)
            G.append(-Lg_h)
            hb.append(Lf_h + min(alpha, 1.0 / float(dt)) * h_val)
    G = np.asarray(G, dtype=dtype).reshape(-1, 3 * n)
    hb = np.asarray(hb, dtype=dtype)
    candidate = nominal_velocity.reshape(-1)
    if len(G):
        try:
            candidate = solve_qp(
                sparse.eye(3 * n, format="csc"),
                -nominal_velocity.reshape(-1).astype(np.float64),
                sparse.csc_matrix(G, dtype=np.float64), hb.astype(np.float64),
                solver="osqp",
            )
        except (SolverError, OSQPException):
            candidate = None
    if candidate is None:
        candidate = np.full(3 * n, np.nan, dtype=dtype)
    # In Crazyflow this casts the solution back to the simulator precision first.
    candidate = np.asarray(candidate, dtype=dtype)
    residual = np.max(G @ candidate - hb, initial=-np.inf)
    threshold = tol * (1.0 + np.max(np.abs(hb), initial=0.0))
    fallback = not (np.all(np.isfinite(candidate)) and residual <= threshold)
    velocity = np.zeros_like(nominal_velocity) if fallback else candidate.reshape(n, 3)
    return velocity, dict(residual=float(residual), threshold=float(threshold), fallback=fallback)


def self_correct(n_retries: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Create a decorator that retries a function n times if it fails."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(self: AppBackend, *args: P.args, **kwargs: P.kwargs) -> R:
            assert isinstance(self, AppBackend), "self_correct decorator must be used on AppBackend"
            try:
                return fn(self, *args, **kwargs)
            except LLMException as e:
                error_message = str(e)
                for i in range(n_retries):
                    try:
                        logger.info("Reprompting due to LLM error")
                        message = "The provided response failed with the following error:"
                        message += f"\n{error_message}\n\n"
                        message += "Analyze the error, re-read the instructions and try again."
                        return self.reprompt.__wrapped__(self, message)
                    except LLMException as inner_e:
                        if i == n_retries - 1:
                            raise inner_e
                        error_message = str(inner_e)
                        continue
                raise e

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess-based visualizer
#
# MuJoCo/OpenGL non supporta reinizializzazione nello stesso processo dopo
# sim.close(). La soluzione è eseguire ogni sessione del visualizzatore in un
# NUOVO subprocess figlio che parte da zero.
#
# Comunicazione: le splines vengono serializzate con pickle in un file
# temporaneo che il subprocess legge. Nessun fork (usiamo subprocess.Popen
# con python -c), quindi non ereditiamo lo stato OpenGL del padre.
# ─────────────────────────────────────────────────────────────────────────────

# Script Python che gira nel subprocess figlio.
# Viene passato come stringa a `python -c "..."`.
_VIZ_SCRIPT = """
from pathlib import Path
import json
import sys, pickle, logging
logging.basicConfig(level=logging.WARNING)

data_path = sys.argv[1]
with open(data_path, "rb") as f:
    payload = pickle.load(f)

splines   = payload["splines"]    # dict drone -> list of scipy splines
settings  = payload["settings"]
duration  = payload["duration"]

output_dir = Path(payload["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

CBF_MODE = payload.get("cbf_mode", "osqp")
if CBF_MODE not in ("off", "osqp"):
    raise ValueError(f"Unknown CBF mode: {CBF_MODE}")
ENABLE_CBF = CBF_MODE != "off"
print(f"[CBF MODE] {CBF_MODE.upper()}", flush=True)

from collections import deque
import time
import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from swarm_gpt.utils.utils import draw_line
from swarm_gpt.core.backend import correct_velocity


from functools import partial
from crazyflow.sim.sim import (
    generate_control_fn,
    generate_wrench_fn,
    generate_physics_fn,
    generate_sync_fn,
)

# ------------for testing------------

def compute_safety_metrics(pos, d_safe=0.20, kz=2.5):
    # pos: (N, 3), posizioni dei droni in metri
    n = pos.shape[0]
    if n < 2:
        return np.inf, np.inf, 0
    ii, jj = np.triu_indices(n, k=1)

    diff = pos[ii] - pos[jj]

    # Distanza euclidea
    distances = np.linalg.norm(diff, axis=1)

    # Stessa barriera utilizzata nel tuo QP
    q = np.array([1.0, 1.0, 1.0 / kz**2])
    h = np.sum(q * diff**2, axis=1) - d_safe**2

    return (
        np.min(h),
        np.min(distances),
        np.sum(h < 0),
    )
# ------------end------------


class CBFSim(Sim):
    # These settings are fixed for each instance, before building the JAX functions.
    cbf_d_safe = 0.20
    cbf_kz = 2.5
    cbf_alpha = 4.0
    cbf_tol = 1e-4

    def __init__(self, *args, cbf_mode="osqp", **kwargs):
        if cbf_mode not in ("off", "osqp"):
            raise ValueError(f"Unknown CBF mode: {cbf_mode}")
        self.cbf_mode = cbf_mode
        self.last_cbf_diagnostics = None
        super().__init__(*args, **kwargs)

    def build_step_fn(self):
        control_fn = generate_control_fn(self.control)
        wrench_fn = generate_wrench_fn(self.physics)
        disturbance_fn = self.disturbance_fn
        self._original_integrate = jax.jit(generate_physics_fn(self.physics, self.integrator))
        sync_fn = generate_sync_fn(self.physics)

        @jax.jit
        def control_and_wrench(data):
            return wrench_fn(control_fn(data))

        def single_step(data, _):
            data = control_and_wrench(data)
            data = disturbance_fn(data)
            data, diagnostics = self._integrate(data)
            data = data.replace(core=data.core.replace(steps=data.core.steps + 1))
            data = sync_fn(data)
            diagnostics = dict(diagnostics, positions=data.states.pos,
                               velocities=data.states.vel,
                               steps=data.core.steps.reshape(self.n_worlds))
            return data, diagnostics

        @partial(jax.jit, static_argnames="n_steps")
        def step_jax(data, n_steps=1):
            return jax.lax.scan(single_step, data, length=n_steps)

        def step(data, n_steps=1):
            optimize_mjx_model = data.mjx_model is None
            if optimize_mjx_model:
                data = data.replace(mjx_model=self.mjx_model)
            data = self.sync_sim2mjx(data)

            if self.cbf_mode == "osqp":
                # Carry the ENTIRE corrected SimData, including controller memory,
                # from one physical substep into the next. References stay at 80 Hz.
                diagnostics = []
                for _ in range(n_steps):
                    data, sample = single_step(data, None)
                    diagnostics.append(sample)
                diagnostics = jax.tree.map(lambda *xs: jnp.stack(xs), *diagnostics)
            else:
                data, diagnostics = step_jax(data, n_steps=n_steps)
            if optimize_mjx_model:
                data = data.replace(mjx_model=None)
            self.last_cbf_diagnostics = diagnostics
            return data

        self._step = step

    def _integrate(self, data):
        dt = 1.0 / data.core.freq
        predicted = self._original_integrate(data)
        nominal_vel = predicted.states.vel

        if self.cbf_mode == "osqp":
            safe_vel, diagnostics = self.correct_velocity_osqp(
                data, nominal_vel, dt
            )
        else:
            safe_vel = nominal_vel
            diagnostics = dict(residual=jnp.full((self.n_worlds,), jnp.nan),
                               threshold=jnp.full((self.n_worlds,), jnp.nan),
                               fallback=jnp.zeros((self.n_worlds,), dtype=bool))

        result = predicted.replace(
            states=predicted.states.replace(
                pos=data.states.pos + safe_vel * dt,
                vel=safe_vel,
            )
        )
        diagnostics = dict(diagnostics, nominal_velocities=nominal_vel)
        return result, diagnostics

    def correct_velocity_osqp(self, data, nominal_vel, dt):
        velocities, diagnostics = [], []
        for p, u_nom in zip(data.states.pos, nominal_vel):
            safe, sample = correct_velocity(np.asarray(p), np.asarray(u_nom), dt)
            velocities.append(jnp.asarray(safe, dtype=u_nom.dtype))
            diagnostics.append(sample)
        safe = jnp.stack(velocities)
        diagnostics = jax.tree.map(lambda *xs: jnp.stack(xs), *diagnostics)
        return safe, diagnostics

fps          = 60
amswarm_freq = settings["axswarm"]["freq"]

sim = CBFSim(
    cbf_mode=CBF_MODE,
    n_worlds=1,
    n_drones=len(splines),
    physics=Physics.analytical,
    control=Control.state,
    freq=settings["sim_freq"],
    attitude_freq=settings["attitude_freq"],
    state_freq=settings["state_freq"],
    device="cpu",
)
sim.max_visual_geom = 100_000
sim.reset()
sim.state_control(np.random.default_rng(0).random((1, sim.n_drones, 13)))
sim.step(sim.freq // sim.control_freq)
jax.block_until_ready(sim.data)
sim.reset()
assert sim.freq % sim.control_freq == 0
reference_freq = payload.get("reference_freq") or sim.control_freq
if reference_freq <= 0 or sim.control_freq % reference_freq != 0:
    raise ValueError("Reference frequency must divide the controller frequency")
reference_stride = int(sim.control_freq / reference_freq)

vel_splines = {k: [s.derivative() for s in v] for k, v in splines.items()}
pos = np.array([[s(0) for s in splines[j]] for j in splines])[None, ...]
if payload.get("initial_positions") is not None:
    pos = np.asarray(payload["initial_positions"])[None, ...]
sim.data = sim.data.replace(
    states=sim.data.states.replace(pos=sim.data.states.pos.at[...].set(pos))
)

rng   = np.random.default_rng(0)
rgbas = rng.random((sim.n_drones, 4))
rgbas[..., 3] = 1
swarm_pos = [deque(maxlen=100) for _ in range(sim.n_drones)]

# Experiment recording of the displayed simulation (None unless the parent set a record dir)
from swarm_gpt.recording.sim_trace import VizTrace
trace = VizTrace.from_env(sim, splines, duration)

tstart   = time.time()
n_steps  = int(duration * sim.control_freq)

# ------------for cbf comparison------------
safety_times = []
safety_positions = []
safety_min_h = []
safety_min_distance = []
safety_violations = []
physics_logs = []
step_wall_seconds = []
# ------------

# --- Diagnostica temporanea: misura dove va il tempo reale del loop -------
spline_time   = 0.0
step_time     = 0.0
draw_time     = 0.0
mjrender_time = 0.0
sleep_time    = 0.0
render_count  = 0
loop_wall_start = time.time()
# ----------------------------------------------------------------------------

try:
    for i in range(n_steps):
        ct = i / sim.control_freq

        _t0 = time.perf_counter()
        if i % reference_stride == 0:
            des_pos = np.array([[s(ct) for s in splines[j]] for j in splines])
            des_vel = np.array([[s(ct) for s in vel_splines[j]] for j in splines])
            controls = np.concatenate(
                (des_pos, des_vel, np.zeros((sim.n_drones, 7))), axis=-1
            )[None, ...]
        spline_time += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        if i % reference_stride == 0:
            sim.state_control(controls)
        sim.step(sim.freq // sim.control_freq)
        jax.block_until_ready((sim.data, sim.last_cbf_diagnostics))
        step_wall_seconds.append(time.perf_counter() - _t0)
        physics_logs.append(jax.tree.map(np.asarray, sim.last_cbf_diagnostics))
        # ------------ cbf comparison ------------
        actual_pos = np.asarray(
            sim.data.states.pos[0]
        ).copy()

        min_h, min_distance, violations = compute_safety_metrics(
            actual_pos,
            d_safe=0.20,
            kz=2.5,
        )
        # ------------

        safety_times.append((i + 1) / sim.control_freq)
        safety_positions.append(actual_pos)
        safety_min_h.append(min_h)
        safety_min_distance.append(min_distance)
        safety_violations.append(violations)

        step_time += time.perf_counter() - _t0
        if trace is not None:
            trace.record(ct, controls, sim)

        if payload.get("render", True) and ((i * fps) % sim.control_freq) < fps:
            _t0 = time.perf_counter()
            for j, dq in enumerate(swarm_pos):
                dq.append(np.asarray(sim.data.states.pos[0, j]))
                draw_line(sim, np.array(dq), rgba=rgbas[j % len(rgbas)], min_size=2, max_size=5)
            draw_time += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            sim.render()
            mjrender_time += time.perf_counter() - _t0

            render_count += 1
            dt = ct - (time.time() - tstart)
            if dt > 0:
                time.sleep(dt)
                sleep_time += dt
finally:
    if trace is not None:
        trace.close()
    total_wall = time.time() - loop_wall_start
    print(
        f"[VIZ TIMING] expected_duration={duration:.2f}s actual_wall={total_wall:.2f}s "
        f"n_steps={n_steps} renders={render_count} | "
        f"spline_eval={spline_time:.2f}s sim_step={step_time:.2f}s "
        f"draw_line={draw_time:.2f}s mj_render={mjrender_time:.2f}s slept={sleep_time:.2f}s",
        flush=True,
    )

    # ------------ for cbf comparison ------------

    mode = f"cbf_{CBF_MODE}"

    times = np.asarray(safety_times)
    positions = np.asarray(safety_positions)
    min_h = np.asarray(safety_min_h)
    min_distance = np.asarray(safety_min_distance)
    violations = np.asarray(safety_violations)

    # Each sample below is one physical step (400 Hz), unlike the 80 Hz traces above.
    physics = {
        key: np.concatenate([batch[key] for batch in physics_logs], axis=0)[:, 0]
        for key in ("positions", "velocities", "nominal_velocities", "steps",
                    "residual", "threshold", "fallback")
    } if physics_logs else {}
    physics_metrics = np.asarray([
        compute_safety_metrics(p, sim.cbf_d_safe, sim.cbf_kz)
        for p in physics.get("positions", [])
    ]).reshape(-1, 3)
    physics_fields = {f"physics_{key}": value for key, value in physics.items()}
    physics_fields.update(
        physics_timestamps=physics.get("steps", np.array([])) / sim.freq,
        physics_min_h=physics_metrics[:, 0],
        physics_min_distance=physics_metrics[:, 1],
        physics_violations=physics_metrics[:, 2].astype(int),
    )
    spline_times = np.r_[0.0, physics_fields["physics_timestamps"]]
    spline_positions = np.stack([
        np.stack([s(spline_times) for s in axes], axis=-1) for axes in splines.values()
    ], axis=1)
    spline_metrics = np.asarray([
        compute_safety_metrics(p, sim.cbf_d_safe, sim.cbf_kz) for p in spline_positions
    ])

    np.savez_compressed(
        output_dir / f"{mode}.npz",
        timestamps=times,
        positions=positions,
        min_h=min_h,
        min_distance=min_distance,
        violations=violations,
        enable_cbf=ENABLE_CBF,
        cbf_mode=CBF_MODE,
        step_wall_seconds=np.asarray(step_wall_seconds),
        initial_positions=pos[0],
        offline_fallback_count=payload.get("offline_fallback_count", 0),
        spline_timestamps=spline_times,
        spline_positions=spline_positions,
        spline_min_h=spline_metrics[:, 0],
        spline_violations=spline_metrics[:, 2].astype(int),
        **physics_fields,
    )

    summary = {
        "enable_cbf": ENABLE_CBF,
        "cbf_mode": CBF_MODE,
        "sim_freq_hz": sim.freq,
        "reference_freq_hz": reference_freq,
        "state_freq_hz": sim.control_freq,
        "spline_min_h": float(np.min(spline_metrics[:, 0])),
        "spline_samples_with_violations": int(np.sum(spline_metrics[:, 2] > 0)),
        "cbf_params": {"d_safe": sim.cbf_d_safe, "kz": sim.cbf_kz,
                       "alpha": sim.cbf_alpha, "tol": sim.cbf_tol,
                       "fallback": "zero_velocity"},
        "solver_params": ({"settings": "osqp_defaults"} if CBF_MODE == "osqp" else {}),
        "physics_num_samples": int(len(physics_metrics)),
        "fallback_count": int(np.sum(physics.get("fallback", []))),
        "offline_fallback_count": int(payload.get("offline_fallback_count", 0)),
        "physics_min_h": float(np.min(physics_metrics[:, 0])) if len(physics_metrics) else None,
        "physics_samples_with_violations": int(np.sum(physics_metrics[:, 2] > 0)),
        "step_wall_seconds_total": float(np.sum(step_wall_seconds)),
        "step_timing_scope": "state_control + five physics steps + diagnostics synchronization; "
                             "excludes warmup, spline evaluation, rendering and file writes",
        "num_samples": int(len(times)),
        "min_h": float(np.min(min_h)) if len(min_h) else None,
        "min_distance_m": float(np.min(min_distance)) if len(min_distance) else None,
        "samples_with_violations": int(np.sum(violations > 0)),
        "total_pair_violations": int(np.sum(violations)),
    }

    with open(output_dir / f"{mode}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[SAFETY] {mode} | "
        f"min_h={summary['min_h']} | "
        f"min_distance={summary['min_distance_m']} m | "
        f"violating_samples={summary['samples_with_violations']} | "
        f"fallbacks={summary['fallback_count']} | "
        f"offline_fallbacks={summary['offline_fallback_count']}",
        flush=True,
    )
    # ------------

    sim.close()
"""


class AppBackend:
    """Backend class for the swarm_gpt gradio web app."""

    def __init__(
        self,
        config_file: Path,
        *,
        strict_processing: bool = True,
        strict_drone_match: bool = True,
        model_id: str = "gpt-6-luna",
        use_motion_primitives: bool = False,
    ):
        """Initialize the backend by loading the music files and initializing the choreographer.

        Args:
            config_file: Path to the config file.
            strict_processing: Flag to raise an error on waypoint collisions.
            strict_drone_match: Flag to raise an error when preset drones do not match the current
                swarm.
            model_id: The OpenAI GPT model ID.
            use_motion_primitives: If we want LLM to use motion primitives for choreography
        """
        self.root_path = Path(__file__).resolve().parents[2]
        with open(self.root_path / "swarm_gpt/data/settings.yaml", "r") as f:
            self.settings = yaml.safe_load(f)
        # Initialize drone control elements
        self.waypoints = None # High-level LLM commands
        self.splines: dict = {} # Low-level optimized commands from axswarm
        self.offline_corrected_splines: dict = {}  # smoothing lambda -> drone -> xyz splines
        self.offline_fallback_count = 0
        self._offline_deploy_duration: float | None = None
        self._offline_deploy_waypoints: bytes | None = None
        
        self.drone_controller = DroneController(10) # Controller for the Crazyflie drones
        # Initialize chat elements
        self.choreographer = Choreographer(
            config_file=config_file,
            model_id=model_id,
            use_motion_primitives=use_motion_primitives,
        )
        self.mode: Literal["preset", "real"] = "real"
        # self._preset: None | str = None
        self._strict_processing = strict_processing
        self._strict_drone_match = strict_drone_match
        
        # Subprocess visualizer state
        self._viz_proc: subprocess.Popen | None = None
        self._viz_data_path: str | None = None   # path del file pickle temporaneo

    # Subprocess visualizer helpers 

    def _stop_viz(self) -> None:
        """Terminate the visualizer subprocess and clean up the temp file."""
        if self._viz_proc is not None:
            if self._viz_proc.poll() is None:          # ancora in esecuzione
                logger.info("Terminating visualizer subprocess (pid=%d)…", self._viz_proc.pid)
                self._viz_proc.terminate()
                try:
                    self._viz_proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    logger.warning("Visualizer did not exit — killing.")
                    self._viz_proc.kill()
                    self._viz_proc.wait()
            self._viz_proc = None

        if self._viz_data_path and os.path.exists(self._viz_data_path):
            try:
                os.unlink(self._viz_data_path)
            except OSError:
                pass
            self._viz_data_path = None

    def _start_viz(
        self, splines: dict, duration: float, cbf_mode: Literal["off", "osqp"] = "osqp",
        output_dir: Path | None = None,
        *, reference_freq: int | None = None, initial_positions: np.ndarray | None = None,
        offline_fallback_count: int = 0,
    ) -> None:
        """Kill any existing visualizer, serialize splines to disk, launch new subprocess."""
        if cbf_mode not in ("off", "osqp"):
            raise ValueError(f"Unknown CBF mode: {cbf_mode}")
        self._stop_viz()

        # Serializza le splines in un file temporaneo (pickle supporta oggetti scipy)
        payload = {
            "splines": {k: list(v) for k, v in splines.items()},
            "settings": dict(self.settings),
            "duration": duration,
            "cbf_mode": cbf_mode,
            "output_dir": str(output_dir),
            "reference_freq": reference_freq,
            "initial_positions": initial_positions,
            "offline_fallback_count": offline_fallback_count,
        }
        fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="swarmgpt_viz_")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f)
        except Exception:
            os.unlink(tmp_path)
            raise

        self._viz_data_path = tmp_path

        # Lancia un processo Python FRESCO — nessuna eredità del contesto OpenGL
        env = os.environ.copy()
        self._viz_proc = subprocess.Popen(
            [sys.executable, "-c", _VIZ_SCRIPT, tmp_path],
            env=env,
            # Non bloccare stdout/stderr: lascia che i log appaiano nel terminale
            stdout=None,
            stderr=None,
        )
        logger.info("Visualizer subprocess started (pid=%d).", self._viz_proc.pid)

    # @property
    # def presets(self) -> list[str]:
    #     return [s.name for s in (self.root_path / "swarm_gpt/data/presets").glob("*")]

    # @self_correct(n_retries=2)
    # def initial_prompt(self, text: str, *, response: str | None = None) -> list[dict[str, str]]:
    #     logger.info("Generating initial choreography for: %s", text)
    #     self.choreographer.reset_history()
    #     prompt = self.choreographer.format_initial_prompt(text)

    #     fixed_response = response is not None
    #     preset = False
    #     if response is None:
    #         response = self.choreographer.generate_choreography(prompt)
    #     else:
    #         self.choreographer.messages.append({"role": "assistant", "content": response})

    #     try:
    #         self.waypoints = self.choreographer.response2waypoints(
    #             response, strict=self._strict_processing
    #         )
    #     except LLMException as e:
    #         if preset or fixed_response:
    #             raise RuntimeError("Initial prompt failed") from e
    #         raise e
    #     logger.info("Successfully generated choreography")
    #     return self.choreographer.messages

    @self_correct(n_retries=2)
    def initial_prompt(self, text: str, *, response: str | None = None) -> list[dict[str, str]]:
        logger.info("Generating initial choreography for: %s", text)
        self.choreographer.reset_history()
        agent_one = self.choreographer.analyze_command(text)
        if agent_one['routing'] == "existing_system":
            prompt = self.choreographer.format_initial_prompt(text)

            fixed_response = response is not None
            preset = False
            if response is None:
                response = self.choreographer.generate_choreography(prompt)
            else:
                self.choreographer.messages.append({"role": "assistant", "content": response})

            try:
                self.waypoints = self.choreographer.response2waypoints(
                    response, strict=self._strict_processing
                )
            except LLMException as e:
                if preset or fixed_response:
                    raise RuntimeError("Initial prompt failed") from e
                raise e
            logger.info("Successfully generated choreography")
            return self.choreographer.messages
        else:
            # logger.info("Agent2 code generator")
            # constraints_raw = agent_one.get("constraints", [])
            # if isinstance(constraints_raw, dict):
            #     constraints = [
            #         {
            #             "name": k, "description": str(v)
            #         }
            #         for k, v in constraints_raw.items()
            #         if v is not None
            #     ]
            # else:
            #     constraints = constraints_raw
            # print("DEBUG constraints converted:", constraints)
            # code = self.choreographer.agent2.generate(text, constraints)
            # self.waypoints = self.choreographer.agent2.code_to_waypoints(
            #     code, self.choreographer.starting_pos, n_steps=20
            # )
            # logger.info("Successfully generated choreography")

            missing = agent_one.get("missing", [])
            logger.info("Agent B: generating %d new primitive(s): %s", len(missing), missing)

            nodes = Pipeline().run(missing)

            # inject into running module (immediate availability, no reload)
            for node in nodes:
                exec(node.body, _mp_module.__dict__)  # noqa: S102
                _mp_module.motion_primitives[node.name] = {"n_args": node.n_args}

            write_to_file(nodes, _PRIMITIVES_PATH)
            write_descriptions_to_yaml(nodes)
            register_with_router(nodes, self.choreographer.router)

            # run existing system now that the new primitives are available
            # param "text" to have the original command not just the missings
            prompt = self.choreographer.format_initial_prompt(text)
            response = self.choreographer.generate_choreography(prompt)
            try:
                self.waypoints = self.choreographer.response2waypoints(
                    response, strict=self._strict_processing
                )
            except LLMException as e:
                raise e
            logger.info("Successfully generated choreography after primitive generation")

        return self.choreographer.messages
        

    @self_correct(n_retries=3)
    def reprompt(self, message: str) -> list[dict[str, str]]:
        logger.info("Reprompting with: %s", message)
        if message == "":
            return self.choreographer.messages
        prompt = self.choreographer.format_reprompt(message)
        response = self.choreographer.generate_choreography(prompt)
        self.waypoints = self.choreographer.response2waypoints(
            response, strict=self._strict_processing
        )
        logger.info("Successfully generated choreography")
        return self.choreographer.messages

    def simulate(self, gui: bool = True):
        """Run axswarm, build splines, then launch visualizer in a fresh subprocess."""
        logger.info("Simulating trajectories with axswarm")
        self.splines.clear()
        self.offline_corrected_splines.clear()
        self.offline_fallback_count = 0
        self._offline_deploy_duration = None
        self._offline_deploy_waypoints = None
        if hasattr(self, "sim_results"):
            del self.sim_results
        assert self.waypoints is not None, "Please generate a choreography first"
        simulated_waypoints = pickle.dumps(self.waypoints)

        sim_data = None
        for key, data, total in simulate_axswarm(self.waypoints, self.settings, gui=False):
            if key == "progress":
                yield key, data, total
            else:
                sim_data = data

        assert sim_data is not None, "simulate_axswarm returned no data"

        t = sim_data["timestamps"][::10]
        drone_order = list(self.choreographer.agents.items())
        p_nom = np.stack([sim_data["controls"][:, i, :3] for i, _ in drone_order], axis=1).astype(float)
        p_corr = np.empty(p_nom.shape, dtype=float)
        offline_error = None
        try:
            if (len(t) < 5 or len(t) != len(p_nom) or not np.all(np.isfinite(t))
                    or not np.all(np.isfinite(p_nom)) or np.any(np.diff(t) <= 0)):
                raise ValueError("Offline CBF requires aligned finite points and increasing timestamps")
            p_corr[0] = p_nom[0]
            for k in range(len(t) - 1):
                dt = t[k + 1] - t[k]
                v_nom = (p_nom[k + 1] - p_nom[k]) / dt
                try:
                    v_corr, diagnostics = correct_velocity(p_corr[k], v_nom, dt)
                    self.offline_fallback_count += int(diagnostics["fallback"])
                except Exception as exc:
                    raise ValueError(
                        f"Offline CBF failed at sample {k} -> {k+1}, "
                        f"t={t[k]:.6g} -> {t[k+1]:.6g}: {exc}"
                    ) from exc
                p_corr[k + 1] = p_corr[k] + dt * v_corr
        except Exception as exc:
            offline_error = exc
            logger.error("%s", exc)
        print(
            f"[DEBUG backend] building splines: t.shape={t.shape} "
            f"controls.shape={sim_data['controls'].shape}"
        )
        lam = 0.1
        self.splines.clear()
        for i, drone in self.choreographer.agents.items():
            controls = sim_data["controls"][:, i, :3]
            print(
                f"[DEBUG backend] drone {drone} (idx {i}): first controls={controls[0]} "
                f"last controls={controls[-1]}"
            )
            self.splines[drone] = [
                make_smoothing_spline(t, controls[:, j], lam=lam) for j in range(3)
            ]
        print(f"[DEBUG backend] splines built for drones: {list(self.splines.keys())}")
        if offline_error is None:
            try:
                self.offline_corrected_splines = {
                    smoothing: {
                        drone: [make_smoothing_spline(t, p_corr[:, i, axis], lam=smoothing)
                                for axis in range(3)]
                        for i, (_, drone) in enumerate(drone_order)
                    } for smoothing in (0.0, 0.1)
                }
            except Exception as exc:
                offline_error = ValueError(f"Offline corrected spline fit failed: {exc}")
                logger.error("%s", offline_error)

        if gui:
        #     self._start_viz(self.splines, float(t[-1]))
            duration = float(t[-1])

            output_dir = (
                self.root_path
                / "experiments"
                / "cbf_comparisons"
                / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            for cbf_mode in ("off", "osqp"):
                logger.info("Starting Crazyflow CBF %s", cbf_mode.upper())

                self._start_viz(
                    self.splines,
                    duration,
                    cbf_mode=cbf_mode,
                    output_dir=output_dir,
                )

                rc = self._viz_proc.wait()
                if rc != 0:
                    raise RuntimeError(
                        f"Crazyflow CBF {cbf_mode} failed: {rc}"
                    )

            logger.info("CBF comparison saved to %s", output_dir)
            initial_positions = np.array([[s(0.0) for s in axes] for axes in self.splines.values()])
            runs = [("original_off_10hz", self.splines)]
            if offline_error is None:
                runs.extend((f"offline_corrected_lam_{smoothing}_off_10hz", corrected)
                            for smoothing, corrected in self.offline_corrected_splines.items())
            for name, curves in runs:
                self._start_viz(
                    curves, duration, cbf_mode="off", output_dir=output_dir / name,
                    reference_freq=int(self.drone_controller.freq),
                    initial_positions=initial_positions,
                    offline_fallback_count=(self.offline_fallback_count
                                            if curves is not self.splines else 0),
                )
                rc = self._viz_proc.wait()
                if rc != 0:
                    raise RuntimeError(f"Crazyflow {name} failed: {rc}")
                if curves is self.offline_corrected_splines.get(0.1):
                    self._offline_deploy_duration = duration
                    self._offline_deploy_waypoints = simulated_waypoints
                logger.info("%s metrics saved to %s", name, output_dir / name)

        if offline_error is not None:
            raise RuntimeError(str(offline_error)) from offline_error
        logger.info("Simulation successful")
        return sim_data

    def replay(self) -> None:
        """Replay the last simulation without re-running axswarm."""
        assert self.splines, "Please run Simulate first before replaying."
        assert self.waypoints is not None, "Waypoints missing."
        duration = float(np.max(self.waypoints["time"][:, -1]))
        logger.info("Starting replay (duration=%.2f s)", duration)
        self._start_viz(self.splines, duration)

    def deploy(self, drone_ids: list[int] | None = None, *, spline_set: str = "Original"):
        logger.info("Deploying drones")
        if spline_set == "Original":
            splines = self.splines
            assert splines, "Please run the simulation first!"
            duration = float(np.max(self.waypoints["time"][:, -1]))
        elif spline_set == "Offline CBF":
            splines = self.offline_corrected_splines.get(0.1)
            if (not splines or self._offline_deploy_duration is None
                    or self._offline_deploy_waypoints != pickle.dumps(self.waypoints)):
                raise ValueError(
                    "Offline CBF is unavailable: complete its lam=0.1 simulation "
                    "for the current choreography before deploying."
                )
            duration = self._offline_deploy_duration
        else:
            raise ValueError(f"Unknown spline set: {spline_set}")
        if not self.drone_controller._ros_running:
            raise RuntimeError("ROS is not running. Please start ROS before deploying.")

        if drone_ids is not None:
            cfs = self.drone_controller.swarm.allcfs.crazyfliesById
            cfs = {k: v for k, v in cfs.items() if k in drone_ids}
            self.drone_controller.swarm.allcfs.crazyfliesById = cfs

        # for i, drone in enumerate(self.drone_controller.swarm.allcfs.crazyfliesById.values()):
        #     drone.setLEDColor(*colors[i % len(colors)])

        # original_song = self.music_manager.song
        print(self.waypoints)
        
        # duration = next(iter(self.waypoints.values()))[-1, 0]
        logger.info("Deploy spline set: %s (duration=%.2f s)", spline_set, duration)
        # try:
        #     self.music_manager.song = original_song + "[deploy]"
        # except AssertionError:
        #     pass
        self.drone_controller.arm(True)
        self.drone_controller.takeoff(target_height=1.0, duration=3.0)
        self.drone_controller.move_to_initial_positions(height=1.0, duration=3.0)
        #self.music_manager.play()
        # self.drone_controller.run_spline_trajectories(self.splines, duration=duration)
        try:
            self.drone_controller.run_spline_trajectories(splines, duration=duration)
        finally:
            self.drone_controller.land_full_state(landing_height=0.05, duration=6.0)
        
        # self.drone_controller.land()
        # self.music_manager.song = original_song
        logger.info("Deployment successful")

    def load_preset(self, preset_id: str) -> list[dict[str, str]]:
        assert preset_id, "Please select a valid preset"
        assert preset_id in self.presets, "No preset for this song"
        preset_path = self.root_path / "swarm_gpt/data/presets" / preset_id
        n_drones = self.choreographer.num_drones
        preset_n_drones = int(preset_id.split("|")[1].strip())
        if preset_n_drones != n_drones and self._strict_drone_match:
            raise ValueError(
                f"Preset n_drones ({preset_n_drones}) do not match current swarm ({n_drones})"
            )
        with open(preset_path / "history.json", "r") as f:
            history = json.load(f)
        with open(preset_path / "meta.json", "r") as f:
            meta = json.load(f)
        if meta["use_motion_primitives"] != self.choreographer.use_motion_primitives:
            raise ValueError("Preset was generated with a different use_motion_primitives setting")
        assert history[-1]["role"] == "assistant"
        self.choreographer.messages = history
        return history[-1]["content"]

    def save_preset(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preset_name = self.music_manager.song + f" | {self.choreographer.num_drones} | {timestamp}"
        path = self.root_path / "swarm_gpt/data/presets" / preset_name
        path.mkdir(parents=True, exist_ok=True)
        if not self.choreographer.messages:
            raise ValueError("No preset to save. Run Simulation first")
        with open(path / "history.json", "w") as f:
            json.dump(self.choreographer.messages, f)
        meta = {"n_drones": self.choreographer.num_drones, "song": self.music_manager.song}
        meta["use_motion_primitives"] = self.choreographer.use_motion_primitives
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        if self.waypoints is not None:
            np.save(path / "waypoints.npy", self.waypoints)

    def reset_data(self) -> None:
        """Terminate visualizer subprocess, then reset all session state."""
        logger.info("Resetting backend data for a new command...")

        # Termina il subprocess PRIMA di liberare i dati
        self._stop_viz()

        self.waypoints = None
        self.splines.clear()
        self.offline_corrected_splines.clear()
        self.offline_fallback_count = 0
        self._offline_deploy_duration = None
        self._offline_deploy_waypoints = None
        for attr in ("full_trajectory", "sim_results"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.choreographer.reset_history()
        logger.info("Backend reset complete.")
