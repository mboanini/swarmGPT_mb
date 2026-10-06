"""Simulation module for swarm_gpt.

Before we deploy the choreography to the drones, we run a simulation to check if the modified paths
from AMSwarm are collision-free and can be executed. While there is no guarantee that the
trajectories work in reality, it is a good sanity check to ensure that the drones do not crash into
each other or have to perform infeasible maneuvers.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING

import jax
import numpy as np
import matplotlib.pyplot as plt
from axswarm import SolverData, SolverSettings, solve
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from tqdm import tqdm

from swarm_gpt.utils import MusicManager
from swarm_gpt.utils.utils import draw_line

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swarm_gpt.utils import MusicManager


def _pairwise_safety_metrics(
    positions: np.ndarray,
    collision_envelope: np.ndarray,
) -> tuple[float, float, float]:
    """Compute minimum pairwise safety metrics.

    Args:
        positions: Drone positions with shape (n_drones, 3).
        collision_envelope: AxSwarm collision envelope [rx, ry, rz].

    Returns:
        min_euclidean: Minimum Euclidean inter-drone distance [m].
        min_normalized: Minimum envelope-normalized distance. Safe boundary is 1.0.
        h_min: Minimum barrier-like value h = d_normalized^2 - 1. Safe iff h >= 0.
    """
    n_drones = positions.shape[0]
    if n_drones < 2:
        return np.inf, np.inf, np.inf
    relative_positions = positions[:, None, :] - positions[None, :, :]
    euclidean = np.linalg.norm(relative_positions, axis=-1)
    normalized = np.linalg.norm(relative_positions / collision_envelope, axis=-1)
    off_diag = ~np.eye(n_drones, dtype=bool)
    min_euclidean = float(euclidean[off_diag].min())
    min_normalized = float(normalized[off_diag].min())
    h_min = min_normalized**2 - 1
    return min_euclidean, min_normalized, h_min


def simulate_axswarm(
    waypoints: dict[str, NDArray], settings: dict, gui: bool = False
) -> dict[int, NDArray]:
    """Run the crazyflow simulation from waypoints.

    Args:
        waypoints: The waypoints to fly to. Dictionary of drone IDs to waypoints. Each waypoint
            consists of [time, x, y, z, vx, vy, vz].
        settings: Settings for the simulation and AMSwarm.
        gui: Flag to render the simulation.

    Returns:
        A collection of data from the simulation.
    """

    # =============================================================================
    # THESIS INSTRUMENTATION - PHASE 1
    # Observational logging only: compares AxSwarm's own prediction against the state actually
    # produced by crazyflow. Does not change any ORIGINAL SWARMGPT BEHAVIOR below - pos/vel fed
    # into the next solve() still come from solver_data.u_pos/u_vel, never from sim.data.states,
    # exactly as in the original baseline.
    # =============================================================================
    prediction_times = []
    predicted_next_pos_log = []
    actual_pos_log = []
    prediction_error_log = []
    min_predicted_euclidean_one_step_log = []
    min_actual_euclidean_log = []
    min_predicted_normalized_one_step_log = []
    min_actual_normalized_log = []
    h_min_predicted_one_step_log = []  # worst margin predicted ONE MPC step ahead (data.pos[:,1])
    h_min_actual_log = []
    horizon_times_log = []
    h_min_predicted_horizon_log = []  # worst margin anywhere in the full MPC horizon (data.pos)
    # Prediction generated at the previous AxSwarm solve.
    # data.pos[:, 1] predicts the state one AxSwarm period ahead.
    pending_prediction = None
    pending_prediction_time = None

    # Set up the simulation
    sim = Sim(
        n_worlds=1,
        n_drones=waypoints["pos"].shape[0],
        physics=Physics.analytical,
        control=Control.state,
        freq=settings["sim_freq"],
        attitude_freq=settings["attitude_freq"],
        state_freq=settings["state_freq"],
        device="cpu",
    )
    fps = 60
    sim.max_visual_geom = 100_000

    # JIT compile the simulation
    sim.reset()
    sim.state_control(np.random.random((sim.n_worlds, sim.n_drones, 13)))
    sim.step(sim.freq // sim.control_freq)
    sim.reset()

    # Set up solver
    solver_settings = {
        k: v if not isinstance(v, list) else np.asarray(v) for k, v in settings["axswarm"].items()
    }
    solver_settings = SolverSettings(**solver_settings)
    dynamics = settings["Dynamics"]
    A, B = np.asarray(dynamics["A"]), np.asarray(dynamics["B"])
    A_prime, B_prime = np.asarray(dynamics["A_prime"]), np.asarray(dynamics["B_prime"])
    solver_data = SolverData.init(
        waypoints=waypoints,
        K=solver_settings.K,
        N=solver_settings.N,
        A=A,
        B=B,
        A_prime=A_prime,
        B_prime=B_prime,
        freq=solver_settings.freq,
        smoothness_weight=solver_settings.smoothness_weight,
        input_smoothness_weight=solver_settings.input_smoothness_weight,
        input_continuity_weight=solver_settings.input_continuity_weight,
    )
    n_steps = int(waypoints["time"][0, -1] * sim.control_freq)
    solve_every_n_steps = sim.control_freq // solver_settings.freq

    assert sim.freq % sim.control_freq == 0, (
        "control freq {sim.control_freq} must be divisible by sim.freq {sim.freq}"
    )
    assert sim.control_freq % solver_settings.freq == 0, (
        "control freq {sim.control_freq} must be divisible by amswarm freq {solver_settings.freq}"
    )

    # Set up initial states
    control = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
    pos = sim.data.states.pos.at[0, ...].set(waypoints["pos"][:, 0])
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    pos, vel = np.asarray(sim.data.states.pos[0]), np.asarray(sim.data.states.vel[0])
    states, controls, solve_times = [], [], []  # logging variables

    # Set up colours for tracking lines
    rng = np.random.default_rng(0)
    rgbas = rng.random((sim.n_drones, 4))
    rgbas[..., 3] = 1

    tstart = time.time()
    for step in tqdm(range(n_steps)):
        yield "progress", step + 1, n_steps
        t = step / sim.control_freq
        if step % solve_every_n_steps == 0:
            # ---------------------------------------------------------
            # PHASE 1: validate the prediction generated at the PREVIOUS AxSwarm solve against
            # the current Crazyflow state. pending_prediction corresponds to data.pos[:, 1]
            # predicted one MPC period ago. Reading-only: does not change pos/vel below.
            # ---------------------------------------------------------
            if pending_prediction is not None:
                time_error = abs(t - pending_prediction_time)
                if time_error > 1e-6:
                    logger.warning(
                        f"[PHASE1] Prediction timing mismatch: "
                        f"expected={pending_prediction_time:.6f}s, actual={t:.6f}s"
                    )
                actual_pos = np.asarray(sim.data.states.pos[0]).copy()
                prediction_error = np.linalg.norm(actual_pos - pending_prediction, axis=-1)
                min_actual_euclidean, min_actual_normalized, h_min_actual = (
                    _pairwise_safety_metrics(
                        actual_pos, np.asarray(solver_settings.collision_envelope)
                    )
                )
                (
                    min_predicted_euclidean_one_step,
                    min_predicted_normalized_one_step,
                    h_min_predicted_one_step,
                ) = _pairwise_safety_metrics(
                    pending_prediction, np.asarray(solver_settings.collision_envelope)
                )

                prediction_times.append(t)
                predicted_next_pos_log.append(pending_prediction)
                actual_pos_log.append(actual_pos)
                prediction_error_log.append(prediction_error)
                min_predicted_euclidean_one_step_log.append(min_predicted_euclidean_one_step)
                min_actual_euclidean_log.append(min_actual_euclidean)
                min_predicted_normalized_one_step_log.append(min_predicted_normalized_one_step)
                min_actual_normalized_log.append(min_actual_normalized)
                h_min_predicted_one_step_log.append(h_min_predicted_one_step)
                h_min_actual_log.append(h_min_actual)

                if len(prediction_times) % 20 == 1:
                    print(
                        f"[PHASE1] t={t:.3f}s | mean prediction error="
                        f"{np.mean(prediction_error):.4f} m | "
                        f"max error={np.max(prediction_error):.4f} m | "
                        f"h_pred_1step={h_min_predicted_one_step:.4f} | h_actual={h_min_actual:.4f}"
                    )

            state = np.concat((pos, vel), axis=-1)
            t_solve = time.perf_counter()

            # PHASE 1 diagnostic:
            # AxSwarm computes its collision-activation distances from the
            # trajectories stored BEFORE the current solve.
            pre_solve_pos = np.asarray(solver_data.pos).copy()

            pre_solve_min_normalized = np.full(
                (sim.n_drones, sim.n_drones),
                np.inf,
            )

            collision_envelope = np.asarray(solver_settings.collision_envelope)

            for i in range(sim.n_drones):
                for j in range(i + 1, sim.n_drones):
                    relative_positions = pre_solve_pos[i] - pre_solve_pos[j]

                    normalized_distances = np.linalg.norm(
                        relative_positions / collision_envelope,
                        axis=-1,
                    )

                    min_normalized = float(np.min(normalized_distances))

                    pre_solve_min_normalized[i, j] = min_normalized
                    pre_solve_min_normalized[j, i] = min_normalized

            success, _, solver_data = solve(state, t, solver_data, solver_settings)
            jax.block_until_ready(solver_data)
            solve_times.append(time.perf_counter() - t_solve)

            # ---------------------------------------------------------
            # Store the one-MPC-step-ahead prediction for the next PHASE 1 check.
            # data.pos[:, 0] = current state x_0 (tautological: the first block of S_x is the
            # identity, input cannot affect step 0). data.pos[:, 1] = predicted state at
            # t + 1/f_axswarm, the point actually comparable to Crazyflow at the next solve.
            # ---------------------------------------------------------
            pending_prediction = np.asarray(solver_data.pos[:, 1]).copy()
            pending_prediction_time = t + 1.0 / solver_settings.freq

            # ---------------------------------------------------------
            # THESIS INSTRUMENTATION - PHASE 1
            # Find the worst pairwise safety margin over the full
            # AxSwarm predicted horizon and record where it occurs.
            # Observational only: does not modify solver behavior.
            # ---------------------------------------------------------
            predicted_horizon = np.asarray(solver_data.pos).copy()
            collision_envelope = np.asarray(
                solver_settings.collision_envelope
            )

            worst_h = np.inf
            worst_k = None
            worst_pair = None
            worst_euclidean = None
            worst_normalized = None

            n_drones = predicted_horizon.shape[0]

            for k in range(predicted_horizon.shape[1]):
                positions_k = predicted_horizon[:, k, :]

                for i in range(n_drones):
                    for j in range(i + 1, n_drones):
                        relative_position = positions_k[i] - positions_k[j]

                        euclidean_distance = np.linalg.norm(
                            relative_position
                        )

                        normalized_distance = np.linalg.norm(
                            relative_position / collision_envelope
                        )

                        h_ij = normalized_distance**2 - 1.0

                        if h_ij < worst_h:
                            worst_h = float(h_ij)
                            worst_k = k
                            worst_pair = (i, j)
                            worst_euclidean = float(euclidean_distance)
                            worst_normalized = float(normalized_distance)

            horizon_times_log.append(t)
            h_min_predicted_horizon_log.append(worst_h)

            # Print only when the predicted horizon contains an
            # envelope violation.
            if worst_h < 0.0:
                future_time = t + worst_k / solver_settings.freq

                i, j = worst_pair

                pre_min_normalized = pre_solve_min_normalized[i, j]

                would_be_active = pre_min_normalized <= 1.0

                print(
                    f"[PHASE1-HORIZON] solve_t={t:.3f}s | "
                    f"k={worst_k} | "
                    f"future_t={future_time:.3f}s | "
                    f"pair=({worst_pair[0]}, {worst_pair[1]}) | "
                    f"distance={worst_euclidean:.6f} m | "
                    f"normalized={worst_normalized:.6f} | "
                    f"h={worst_h:.6f} | "
                    f"pre_solve_min_norm={pre_min_normalized:.6f} | "
                    f"pre_solve_within_envelope={would_be_active}"
                )

            if not all(success):
                logger.info("Solve failed")

            solver_data = solver_data.step(solver_data)

            # ORIGINAL SWARMGPT BEHAVIOR: the next solve() is seeded from AxSwarm's own previous
            # output, never from the measured crazyflow state - the loop is NOT closed here.
            pos, vel = solver_data.u_pos[:, 0], solver_data.u_vel[:, 0]

            control[0, :, :3] = solver_data.u_pos[:, 0]
            control[0, :, 3:6] = solver_data.u_vel[:, 0]

            # Log inputs
            controls.append(control[0, :, :6].copy())
            # ORIGINAL SWARMGPT BEHAVIOR: `states` stores the commanded state reference, not the
            # measured crazyflow state. Kept unchanged in Phase 1 to preserve the original
            # baseline - see actual_pos_log above for the measured crazyflow state.
            states.append(control[0, :, :6].copy())

        # Run the simulation
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)

        # Render simulation with visualizations of the planned trajectories
        if ((step * fps) % sim.control_freq) < fps and gui:
             for i in range(sim.n_drones):
                 draw_line(sim, solver_data.u_pos[i, :], rgba=rgbas[i % len(rgbas)])
             sim.render()
             if (dt := t - (time.time() - tstart)) > 0:
                 time.sleep(dt)
    sim.close()

    # --- PHASE 1: grafici (prediction error, safety margin, min distance) ---
    if len(prediction_times) > 0:
        prediction_errors = np.asarray(prediction_error_log)
        try:
            plt.figure(figsize=(12, 6))
            for drone in range(sim.n_drones):
                plt.plot(prediction_times, prediction_errors[:, drone], label=f"Drone {drone}")
            plt.xlabel("Time [s]")
            plt.ylabel("One-step prediction error [m]")
            plt.title("AxSwarm one-step prediction error")
            plt.grid(True)
            plt.legend()
            plt.savefig("phase1_prediction_error.png")
            plt.close()
            print("[PHASE1] Grafico prediction error salvato in: phase1_prediction_error.png")
        except Exception as e:
            print(f"[PHASE1] Errore nel grafico prediction error: {e}")

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(prediction_times, h_min_predicted_one_step_log, label="AxSwarm predicted (1 step)")
            plt.plot(horizon_times_log, h_min_predicted_horizon_log, ":", label="AxSwarm predicted (full horizon, worst case)")
            plt.plot(prediction_times, h_min_actual_log, label="Crazyflow actual")
            plt.axhline(y=0.0, linestyle="--", label="Safety boundary")
            plt.xlabel("Time [s]")
            plt.ylabel("Minimum h")
            plt.title("Predicted vs simulated safety margin")
            plt.grid(True)
            plt.legend()
            plt.savefig("phase1_safety_margin.png")
            plt.close()
            print("[PHASE1] Grafico safety margin salvato in: phase1_safety_margin.png")
        except Exception as e:
            print(f"[PHASE1] Errore nel grafico safety margin: {e}")

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(
                prediction_times, min_predicted_euclidean_one_step_log, label="AxSwarm predicted (1 step)"
            )
            plt.plot(prediction_times, min_actual_euclidean_log, label="Crazyflow actual")
            plt.xlabel("Time [s]")
            plt.ylabel("Minimum inter-drone distance [m]")
            plt.title("Predicted vs simulated minimum separation")
            plt.grid(True)
            plt.legend()
            plt.savefig("phase1_min_distance.png")
            plt.close()
            print("[PHASE1] Grafico min distance salvato in: phase1_min_distance.png")
        except Exception as e:
            print(f"[PHASE1] Errore nel grafico min distance: {e}")
    # --------------------------------------

    states_array = np.stack(states) if len(states) > 0 else np.zeros((1, sim.n_drones, 6))
    controls_array = np.stack(controls) if len(controls) > 0 else np.zeros((1, sim.n_drones, 6))

    # =============================================================================
    # THESIS INSTRUMENTATION - PHASE 1 SUMMARY
    # Numerical summary of the observational metrics collected above.
    # Does not modify planning or control behavior.
    # =============================================================================
    if len(prediction_times) > 0:
        prediction_errors_arr = np.asarray(prediction_error_log)

        h_pred_1step_arr = np.asarray(h_min_predicted_one_step_log)
        h_actual_arr = np.asarray(h_min_actual_log)
        h_pred_horizon_arr = np.asarray(h_min_predicted_horizon_log)

        d_pred_arr = np.asarray(min_predicted_euclidean_one_step_log)
        d_actual_arr = np.asarray(min_actual_euclidean_log)

        # Same-time mismatch:
        # positive -> actual execution has MORE safety margin than predicted
        # negative -> actual execution has LESS safety margin than predicted
        h_mismatch = h_actual_arr - h_pred_1step_arr

        # Critical event:
        # AxSwarm predicts the next state as safe, but Crazyflow is actually unsafe.
        predicted_safe_actual_unsafe = np.sum(
            (h_pred_1step_arr >= 0.0) & (h_actual_arr < 0.0)
        )

        # Opposite case: prediction says unsafe, execution is actually safe.
        predicted_unsafe_actual_safe = np.sum(
            (h_pred_1step_arr < 0.0) & (h_actual_arr >= 0.0)
        )

        print("\n========== PHASE 1 SUMMARY ==========")

        print("\nPrediction error:")
        print(f"  Mean: {np.mean(prediction_errors_arr):.6f} m")
        print(f"  Max:  {np.max(prediction_errors_arr):.6f} m")

        print("\nSafety margin h:")
        print(f"  Min predicted (1 step): {np.min(h_pred_1step_arr):.6f}")
        print(f"  Min actual:             {np.min(h_actual_arr):.6f}")
        print(f"  Min predicted horizon:  {np.min(h_pred_horizon_arr):.6f}")

        print("\nMinimum separation:")
        print(f"  Min predicted (1 step): {np.min(d_pred_arr):.6f} m")
        print(f"  Min actual:             {np.min(d_actual_arr):.6f} m")

        print("\nPrediction/execution safety mismatch:")
        print(f"  Min(actual h - predicted h): {np.min(h_mismatch):.6f}")
        print(f"  Max(actual h - predicted h): {np.max(h_mismatch):.6f}")

        print("\nSafety events:")
        print(
            "  Predicted-safe / actual-unsafe: "
            f"{predicted_safe_actual_unsafe}"
        )
        print(
            "  Predicted-unsafe / actual-safe: "
            f"{predicted_unsafe_actual_safe}"
        )

        print("=====================================\n")

    sim_log = {
        "num_drones": sim.n_drones,
        "log_freq": solver_settings.freq,
        "sim_freq": sim.freq,
        "timestamps": np.arange(n_steps) / sim.control_freq,
        "states": states_array,
        "controls": controls_array,
        "waypoints": waypoints,
        "simulation_freq": sim.freq,
        "amswarm_every_n_steps": solve_every_n_steps,
        "solve_times": np.array(solve_times),
        "phase1": {
            "timestamps": np.asarray(prediction_times),
            "predicted_next_pos": np.asarray(predicted_next_pos_log),
            "actual_pos": np.asarray(actual_pos_log),
            "prediction_error": np.asarray(prediction_error_log),
            "min_predicted_euclidean_one_step": np.asarray(min_predicted_euclidean_one_step_log),
            "min_actual_euclidean": np.asarray(min_actual_euclidean_log),
            "min_predicted_normalized_one_step": np.asarray(min_predicted_normalized_one_step_log),
            "min_actual_normalized": np.asarray(min_actual_normalized_log),
            "h_min_predicted_one_step": np.asarray(h_min_predicted_one_step_log),
            "h_min_actual": np.asarray(h_min_actual_log),
            "horizon_timestamps": np.asarray(horizon_times_log),
            "h_min_predicted_horizon": np.asarray(h_min_predicted_horizon_log),
            "h_mismatch": (np.asarray(h_min_actual_log) - np.asarray(h_min_predicted_one_step_log)),
            "predicted_safe_actual_unsafe": int(np.sum((np.asarray(h_min_predicted_one_step_log) >= 0.0) & (np.asarray(h_min_actual_log) < 0.0))),
            "predicted_unsafe_actual_safe": int(np.sum((np.asarray(h_min_predicted_one_step_log) < 0.0) & (np.asarray(h_min_actual_log) >= 0.0))),
        },
    }
    yield "result", sim_log, "placeholder"
    # return sim_log


def simulate_spline(
    splines: dict, settings: dict, t: float, music_manager: MusicManager, gui: bool
):
    """Run the simulation using splines as control reference."""
    # Setting Up Simulation
    fps = 60
    amswarm_freq = settings["axswarm"]["freq"]
    sim = Sim(
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
    # JIT compile the simulation
    sim.reset()
    sim.state_control(np.random.random((1, sim.n_drones, 13)))
    sim.step(sim.freq // sim.control_freq)
    sim.reset()

    vel_splines = {i: [s.derivative() for s in splines[i]] for i in splines}
    assert sim.freq % sim.control_freq == 0, (
        "control freq {sim.control_freq} must be divisible by sim.freq {sim.freq}"
    )
    assert sim.control_freq % amswarm_freq == 0, (
        "control freq {sim.control_freq} must be divisible by amswarm freq {amswarm_freq}"
    )
    # Setting Up Initial States
    pos = np.array([[s(0) for s in splines[j]] for j in splines])[None, ...]
    assert pos.shape == sim.data.states.pos.shape, (
        f"Initial drone position shape mismatch ({pos.shape}) vs ({sim.data.states.pos.shape})"
    )
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[...].set(pos))
    )

    # Set up colours for tracking lines
    rng = np.random.default_rng(0)
    rgbas = rng.random((sim.n_drones, 4))
    rgbas[..., 3] = 1
    swarm_pos = [deque(maxlen=100) for _ in range(sim.n_drones)]
    # Start music if a song is specified
    if music_manager is not None and gui:
        music_manager.play()
        ...

    # MAIN SIMULATION LOOP
    tstart = time.time()
    for i in tqdm(range(0, int(t * sim.control_freq))):
        current_time = i / sim.control_freq
        des_pos = np.array([[s(current_time) for s in splines[j]] for j in splines])
        des_vel = np.array([[s(current_time) for s in vel_splines[j]] for j in splines])
        controls = np.concatenate((des_pos, des_vel, np.zeros((sim.n_drones, 7))), axis=-1)[
            None, ...
        ]
        # Updates Simulation data
        sim.state_control(controls)
        sim.step(sim.freq // sim.control_freq)

        # Set up tracking lines that show the future drone positions
        if (((i * fps) % sim.control_freq) < fps) and gui:
            for j, dq in enumerate(swarm_pos):
                dq.append(np.asarray(sim.data.states.pos[0, j]))
                draw_line(sim, np.array(dq), rgba=rgbas[j % len(rgbas)], min_size=2, max_size=5)

            sim.render()
            if (dt := current_time - (time.time() - tstart)) > 0:
                time.sleep(dt)
    sim.close()
