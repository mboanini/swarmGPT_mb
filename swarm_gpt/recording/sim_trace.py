"""Raw time series of AxSwarm, Crazyflow, the visualizer simulation and the smoothed splines.

Conventions for every array written here:
    - index order (time, drone, ...); drone index i is the Choreographer index, its Crazyflie ID is
      meta["drone_ids"][i] (Choreographer.agents)
    - world frame of the choreography (x, y, z up), meters, m/s, rad/s; quaternions are [x, y, z, w]
    - t_sim: simulation time [s] from the trajectory start (step counter / frequency)
    - t_wall_ns: time.time_ns() when the sample was taken (UTC epoch)

Simulated accelerations are not recorded: crazyflow's states_deriv stays zero after sim.step() with
the analytical physics used here; derive them offline from the 80 Hz velocities.

Data are flushed to <dir>/chunks/ while the simulation runs and merged into one .npz per stream when
it ends, so a crash or kill keeps everything up to the last chunk.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_gpt.recording.recorder import REC, save_npz, to_jsonable

logger = logging.getLogger(__name__)

VIZ_DIR_ENV = "SWARMGPT_VIZ_RECORD_DIR"

AXSWARM_FIELDS = {
    "t_sim": "(S,) s, simulation time of the solve (t passed to axswarm.solve)",
    "t_wall_ns": "(S,) ns, wall clock right after the solve",
    "solve_time_s": "(S,) s, wall-clock duration of solve() incl. block_until_ready",
    "solver_input_state": "(S,n,6) [m, m/s] state passed to solve(): initial state at the first "
    "solve, afterwards the previous solve's u_pos/u_vel (open loop, NOT the measured state)",
    "pred_pos": "(S,n,K+1,3) m, predicted positions solver_data.pos; sample k at t_sim + k/f_mpc",
    "pred_vel": "(S,n,K+1,3) m/s, predicted velocities, from S_x @ x_0 + S_u_W_input @ zeta",
    "pred_pos_reconstruction_error": "(S,) m, max |reconstructed pos - solver_data.pos| (check)",
    "u_pos": "(S,n,K,3) m, input position reference spline samples before step(); u_pos[:,1] "
    "is the reference sent to Crazyflow until the next solve",
    "u_vel": "(S,n,K,3) m/s, input velocity reference samples",
    "u_acc": "(S,n,K,3) m/s^2, input acceleration reference samples",
    "zeta": "(S,n,3(N+1)) spline coefficients of the solution",
    "distance_matrix": "(S,n,n,K+1) AxSwarm pairwise distance matrix used by the solve",
    "success": "(S,n) bool, per-drone solver success",
    "iters": "(S,n) int, per-drone solver iterations",
}

CRAZYFLOW_FIELDS = {
    "t_cmd": "(M,) s, time at which cmd_state was applied: step/f_control",
    "t_sim": "(M,) s, time of the measured state, i.e. after the control step: (step+1)/f_control",
    "t_wall_ns": "(M,) ns, wall clock after the step",
    "sim_steps": "(M,) int, crazyflow physics step counter (t = sim_steps / sim_freq)",
    "cmd_state": "(M,n,13) state command actually passed to sim.state_control for this step "
    "(CBF-filtered if enabled): [pos(3) m, vel(3) m/s, acc(3), yaw, rates(3)]",
    "pos": "(M,n,3) m, measured simulated position",
    "vel": "(M,n,3) m/s, measured simulated velocity",
    "quat": "(M,n,4) measured orientation quaternion [x, y, z, w]",
    "ang_vel": "(M,n,3) rad/s, measured angular velocity",
}

VIZ_FIELDS = {
    "t_cmd": "(M,) s, spline time ct at which the reference cmd_state was evaluated and applied",
    "t_sim": "(M,) s, time of the measured state, after the control step: ct + 1/f_control",
    "t_wall_ns": "(M,) ns, wall clock after the step (the visualizer runs in real time)",
    "cmd_state": "(M,n,13) command from the smoothed splines: [pos(3), vel(3)=spline derivative, 0]",
    "pos": "(M,n,3) m, measured simulated position",
    "vel": "(M,n,3) m/s, measured simulated velocity",
    "quat": "(M,n,4) measured orientation quaternion [x, y, z, w]",
    "ang_vel": "(M,n,3) rad/s, measured angular velocity",
}


class ChunkedLog:
    """Rows of equally shaped arrays, flushed to chunk files and merged into one .npz on close."""

    def __init__(self, directory: Path, name: str, chunk_rows: int):
        self.directory, self.name, self.chunk_rows = Path(directory), name, chunk_rows
        self.rows: dict[str, list] = {}
        self.n_chunks = 0
        self.n_rows = 0

    def append(self, **row: Any) -> None:
        for key, value in row.items():
            self.rows.setdefault(key, []).append(np.array(value, copy=True))
        self.n_rows += 1
        if len(next(iter(self.rows.values()))) >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows or not next(iter(self.rows.values())):
            return
        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        save_npz(self.directory / "chunks" / f"{self.name}_{self.n_chunks:05d}.npz", **arrays)
        self.n_chunks += 1
        self.rows = {}

    def close(self) -> Path | None:
        """Merge all chunks into <dir>/<name>.npz, then delete the chunk files."""
        self.flush()
        chunk_files = sorted((self.directory / "chunks").glob(f"{self.name}_*.npz"))
        if not chunk_files:
            return None
        parts: dict[str, list] = {}
        for path in chunk_files:
            with np.load(path) as data:
                for key in data.files:
                    parts.setdefault(key, []).append(data[key])
        merged = save_npz(self.directory / f"{self.name}.npz",
                          **{k: np.concatenate(v) for k, v in parts.items()})
        if merged is not None:
            for path in chunk_files:
                path.unlink()
            try:
                (self.directory / "chunks").rmdir()
            except OSError:
                pass
        return merged


class SimTrace:
    """Recorder for simulate_axswarm. Created by sim.py only while a simulation is being recorded."""

    def __init__(self, directory: Path, meta: dict):
        self.directory = Path(directory)
        self.meta = meta
        self.solves = ChunkedLog(self.directory, "axswarm_solves", chunk_rows=80)
        self.steps = ChunkedLog(self.directory, "crazyflow_states_80hz", chunk_rows=800)
        self._matrices = None
        self.closed = False
        REC.save_json(self.directory / "meta.json", meta)

    @classmethod
    def start(cls, sim: Any, solver_settings: Any, settings: dict, waypoints: dict,
              enable_cbf: bool, cbf_params: dict | None) -> SimTrace | None:
        """Return a trace if the caller (AppBackend.simulate hook) requested one, else None."""
        directory = getattr(REC, "pending_sim_dir", None)
        if directory is None:
            return None
        REC.pending_sim_dir = None
        try:
            meta = {
                "kind": "axswarm_crazyflow_simulation",
                "n_drones": sim.n_drones,
                "drone_ids": REC.drone_ids,
                "sim_freq_hz": sim.freq,
                "control_freq_hz": sim.control_freq,
                "axswarm_freq_hz": solver_settings.freq,
                "horizon_K": solver_settings.K,
                "spline_order_N": solver_settings.N,
                "collision_envelope_m": np.asarray(solver_settings.collision_envelope),
                "pos_min_m": settings["axswarm"].get("pos_min"),
                "pos_max_m": settings["axswarm"].get("pos_max"),
                "duration_s": float(waypoints["time"][0, -1]),
                "enable_cbf": enable_cbf,
                "cbf_params": cbf_params,
                "settings": settings,
                "initial_state": "crazyflow pos = waypoints['pos'][:, 0], vel = 0, quat = identity",
                "fields": {"axswarm_solves.npz": AXSWARM_FIELDS,
                           "crazyflow_states_80hz.npz": CRAZYFLOW_FIELDS},
                "t_start_wall_ns": time.time_ns(),
            }
            trace = cls(directory, to_jsonable(meta))
            REC.active_trace = trace
            return trace
        except Exception:
            logger.exception("SimTrace: could not start")
            return None

    def record_solve(self, t: float, state: Any, solver_data: Any, success: Any, iters: Any,
                     solve_time: float) -> None:
        try:
            pos = np.asarray(solver_data.pos)
            n, k1 = pos.shape[:2]
            if self._matrices is None:
                m = solver_data.matrices
                self._matrices = (np.asarray(m.S_x), np.asarray(m.S_u_W_input))
            s_x, s_uw = self._matrices
            x0, zeta = np.asarray(solver_data.x_0), np.asarray(solver_data.zeta)
            full = (x0 @ s_x.T + zeta @ s_uw.T).reshape(n, k1, 6)
            self.solves.append(
                t_sim=t,
                t_wall_ns=time.time_ns(),
                solve_time_s=solve_time,
                solver_input_state=np.asarray(state),
                pred_pos=pos,
                pred_vel=full[..., 3:],
                pred_pos_reconstruction_error=np.max(np.abs(full[..., :3] - pos)),
                u_pos=np.asarray(solver_data.u_pos),
                u_vel=np.asarray(solver_data.u_vel),
                u_acc=np.asarray(solver_data.u_acc),
                zeta=zeta,
                distance_matrix=np.asarray(solver_data.distance_matrix),
                success=np.asarray(success, dtype=bool).reshape(n),
                iters=np.asarray(iters).reshape(n),
            )
        except Exception:
            logger.exception("SimTrace: could not record solve")

    def record_step(self, step: int, sent_control: Any, sim: Any) -> None:
        try:
            states = sim.data.states
            self.steps.append(
                t_cmd=step / sim.control_freq,
                t_sim=(step + 1) / sim.control_freq,
                t_wall_ns=time.time_ns(),
                sim_steps=int(np.asarray(sim.data.core.steps).reshape(-1)[0]),
                cmd_state=np.asarray(sent_control)[0],
                pos=np.asarray(states.pos[0]),
                vel=np.asarray(states.vel[0]),
                quat=np.asarray(states.quat[0]),
                ang_vel=np.asarray(states.ang_vel[0]),
            )
        except Exception:
            logger.exception("SimTrace: could not record step")

    def close(self, status: str = "complete", error: dict | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        REC.active_trace = None
        try:
            self.solves.close()
            self.steps.close()
            REC.save_json(self.directory / "status.json",
                          {"status": status, "error": error, "n_solves": self.solves.n_rows,
                           "n_steps": self.steps.n_rows, "t_end_wall_ns": time.time_ns()})
        except Exception:
            logger.exception("SimTrace: could not close")


class VizTrace:
    """Recorder for the visualizer subprocess (Crazyflow driven by the smoothed splines)."""

    def __init__(self, directory: Path, meta: dict):
        self.directory = Path(directory)
        self.log = ChunkedLog(self.directory, "visualizer_states_80hz", chunk_rows=800)
        self.closed = False
        save_npz_meta = {**meta, "fields": {"visualizer_states_80hz.npz": VIZ_FIELDS}}
        REC.save_json(self.directory / "meta.json", save_npz_meta)

    @classmethod
    def from_env(cls, sim: Any, splines: dict, duration: float) -> VizTrace | None:
        """Active only if the parent process set SWARMGPT_VIZ_RECORD_DIR."""
        directory = os.environ.get(VIZ_DIR_ENV)
        if not directory:
            return None
        try:
            trace = cls(Path(directory), to_jsonable({
                "kind": "visualizer_spline_simulation",
                "drone_ids": list(splines.keys()),  # drone index i -> Crazyflie ID
                "n_drones": sim.n_drones,
                "sim_freq_hz": sim.freq,
                "control_freq_hz": sim.control_freq,
                "duration_s": duration,
                "pid": os.getpid(),
                "t_start_wall_ns": time.time_ns(),
            }))
            # _stop_viz() terminates this process: turn SIGTERM into SystemExit so that the
            # script's finally block runs and the recorded data are saved.
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
            return trace
        except Exception:
            logger.exception("VizTrace: could not start")
            return None

    def record(self, ct: float, controls: Any, sim: Any) -> None:
        try:
            states = sim.data.states
            self.log.append(
                t_cmd=ct,
                t_sim=ct + 1.0 / sim.control_freq,
                t_wall_ns=time.time_ns(),
                cmd_state=np.asarray(controls)[0],
                pos=np.asarray(states.pos[0]),
                vel=np.asarray(states.vel[0]),
                quat=np.asarray(states.quat[0]),
                ang_vel=np.asarray(states.ang_vel[0]),
            )
        except Exception:
            logger.exception("VizTrace: could not record step")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.log.close()
            REC.save_json(self.directory / "status.json",
                          {"n_steps": self.log.n_rows, "t_end_wall_ns": time.time_ns()})
        except Exception:
            logger.exception("VizTrace: could not close")


# ---------------------------------------------------------------------- outputs of simulate()

PREDICTION_SAFETY_KEYS = {  # sim_log["phase1"] key -> exported name
    "timestamps": "t_sim",
    "predicted_next_pos": "predicted_pos_one_step",
    "actual_pos": "measured_pos",
    "prediction_error": "prediction_error_m",
    "min_predicted_euclidean_one_step": "min_distance_predicted_one_step_m",
    "min_actual_euclidean": "min_distance_measured_m",
    "min_predicted_normalized_one_step": "min_normalized_distance_predicted_one_step",
    "min_actual_normalized": "min_normalized_distance_measured",
    "h_min_predicted_one_step": "h_min_predicted_one_step",
    "h_min_actual": "h_min_measured",
    "horizon_timestamps": "horizon_t_sim",
    "h_min_predicted_horizon": "h_min_predicted_horizon",
    "h_mismatch": "h_measured_minus_predicted",
}


def save_simulation_outputs(directory: Path, sim_log: dict, start_wall: float) -> None:
    """Save the dict returned by simulate_axswarm with explicit names, plus the generated plots."""
    directory = Path(directory)
    try:
        wp = sim_log["waypoints"]
        save_npz(directory / "waypoints.npz", drone_ids=_drone_id_array(sim_log["num_drones"]),
                 **{k: np.asarray(v) for k, v in wp.items()})
        every = int(sim_log["amswarm_every_n_steps"])
        controls = np.asarray(sim_log["controls"])
        t_ref = np.asarray(sim_log["timestamps"])[::every][: len(controls)]
        save_npz(directory / "axswarm_reference_8hz.npz", t_sim=t_ref,
                 reference_pos=controls[..., :3], reference_vel=controls[..., 3:6],
                 solve_time_s=np.asarray(sim_log["solve_times"]))
        phase1 = sim_log.get("phase1", {})
        arrays = {new: np.asarray(phase1[old]) for old, new in PREDICTION_SAFETY_KEYS.items()
                  if old in phase1}
        if arrays:
            save_npz(directory / "prediction_and_safety.npz", **arrays)
        if "cbf" in sim_log:
            cbf = {k: v for k, v in sim_log["cbf"].items() if k != "params"}
            save_npz(directory / "cbf_filter_80hz.npz", **cbf)
        summary = {
            "num_drones": sim_log["num_drones"],
            "drone_ids": REC.drone_ids,
            "axswarm_freq_hz": sim_log["log_freq"],
            "sim_freq_hz": sim_log["sim_freq"],
            "axswarm_every_n_control_steps": every,
            "n_control_steps": len(sim_log["timestamps"]),
            "solve_time_s": _stats(sim_log["solve_times"]),
            "predicted_safe_measured_unsafe": phase1.get("predicted_safe_actual_unsafe"),
            "predicted_unsafe_measured_safe": phase1.get("predicted_unsafe_actual_safe"),
            "cbf_params": sim_log.get("cbf", {}).get("params"),
            "files": {
                "waypoints.npz": "Choreographer waypoints: time (n,T) s, pos/vel/acc (n,T,3) m",
                "axswarm_reference_8hz.npz": "sim_log['controls'] (== sim_log['states']): the "
                "AxSwarm reference sent to Crazyflow, one row per solve, held until the next one",
                "prediction_and_safety.npz": "one-step AxSwarm prediction vs measured Crazyflow "
                "state at each solve (sim_log['phase1'], renamed)",
            },
        }
        REC.save_json(directory / "summary.json", summary)
        for png in Path.cwd().glob("phase1_*.png"):  # simulate_axswarm writes them to the cwd
            if png.stat().st_mtime >= start_wall:
                (directory / "plots").mkdir(exist_ok=True)
                shutil.copy2(png, directory / "plots" / png.name)
    except Exception:
        logger.exception("Recorder: could not save the simulation outputs")


def save_splines(directory: Path, splines: dict, sim_log: dict) -> None:
    """Save the smoothed splines used by the visualizer and by deploy(), exactly and sampled."""
    try:
        ids = list(splines.keys())
        arrays: dict[str, Any] = {"drone_ids": np.asarray(ids)}
        for cf_id, axes in splines.items():
            for axis, spline in zip("xyz", axes):
                arrays[f"cf{cf_id}_{axis}_knots"] = np.asarray(spline.t)
                arrays[f"cf{cf_id}_{axis}_coefficients"] = np.asarray(spline.c)
                arrays[f"cf{cf_id}_{axis}_degree"] = np.asarray(spline.k)
        every = int(sim_log["amswarm_every_n_steps"])
        controls = np.asarray(sim_log["controls"])
        # backend.simulate fits the splines to sim_log["timestamps"][::10] and controls[:, i, :3]
        fit_t = np.asarray(sim_log["timestamps"])[::10]
        arrays["fit_t_s"] = fit_t
        arrays["fit_pos"] = controls[..., :3]
        t_end = float(fit_t[-1]) if len(fit_t) else 0.0
        t80 = np.arange(0.0, t_end + 1e-9, 1.0 / (every * float(sim_log["log_freq"])))
        arrays["t_80hz"] = t80
        arrays["pos_80hz"] = np.stack(
            [np.stack([s(t80) for s in splines[i]], -1) for i in ids], 1)
        arrays["vel_80hz"] = np.stack(
            [np.stack([s.derivative()(t80) for s in splines[i]], -1) for i in ids], 1)
        save_npz(Path(directory) / "smoothed_splines.npz", **arrays)
        REC.save_json(Path(directory) / "smoothed_splines.json", {
            "description": "scipy make_smoothing_spline(lam=0.1) per drone and axis, fitted in "
            "AppBackend.simulate to the AxSwarm references; evaluated by the visualizer at 80 Hz "
            "and by DroneController.run_spline_trajectories at its rate (velocity = derivative)",
            "rebuild": "scipy.interpolate.BSpline(knots, coefficients, degree)",
            "drone_ids": ids,
            "units": "t s, pos m, vel m/s, world frame",
            "pos_80hz_shape": "(M, n, 3), drone order = drone_ids",
        })
    except Exception:
        logger.exception("Recorder: could not save the splines")


def _drone_id_array(n: int) -> np.ndarray:
    return np.asarray([REC.drone_ids.get(i, -1) for i in range(n)])


def _stats(values: Any) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    return {"n": int(values.size), "mean": float(values.mean()), "max": float(values.max()),
            "min": float(values.min()), "total": float(values.sum())}
