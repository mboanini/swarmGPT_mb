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

    # Inizializzazione liste per il grafico di analisi
    scostamenti_log = []
    tempi_log = []

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
            state = np.concat((pos, vel), axis=-1)
            t_solve = time.perf_counter()
            success, _, solver_data = solve(state, t, solver_data, solver_settings)
            jax.block_until_ready(solver_data)
            solve_times.append(time.perf_counter() - t_solve)

            # --- LOGICA DI CONFRONTO NOMINALE VS SICURO ---
            # 1. Trova il punto ideale nei waypoint originali
            idx_t = np.argmin(np.abs(waypoints["time"][0] - t))
            pos_nominale = waypoints["pos"][:, idx_t] # (n_droni, 3)
            
            # 2. Prendi la posizione calcolata dal solver (quella sicura)
            pos_sicura = solver_data.u_pos[:, 0] # (n_droni, 3)
            
            # 3. Calcola la distanza Euclidea per ogni drone
            distanze = np.linalg.norm(pos_sicura - pos_nominale, axis=-1)
            
            # 4. Salva i dati per il grafico
            scostamenti_log.append(distanze)
            tempi_log.append(t)

            # Stampa live se l'intervento è pesante
            if np.any(distanze > 0.1): 
                logger.warning(f"[SAFETY] t={t:.2f}s: Deviazione max {np.max(distanze):.2f}m")

            if not all(success):
                logger.info("Solve failed")
            # print("sim.py - simulate_axswarm: solver_data = ")
            # print(solver_data)

            solver_data = solver_data.step(solver_data)

            pos, vel = solver_data.u_pos[:, 0], solver_data.u_vel[:, 0]
            # print("sim.py - simulate_axswarm: pos = ")
            # print(pos)
            # print("sim.py - simulate_axswarm: vel = ")
            # print(vel)

            control[0, :, :3] = solver_data.u_pos[:, 0]
            control[0, :, 3:6] = solver_data.u_vel[:, 0]

            # Log inputs
            controls.append(control[0, :, :6].copy())
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

    # --- GENERAZIONE DEL GRAFICO FINALE ---
    try:
        plt.figure(figsize=(12, 6))
        data_plot = np.array(scostamenti_log) # Shape: (steps, n_droni)
        for i in range(sim.n_drones):
            plt.plot(tempi_log, data_plot[:, i], label=f'Drone {i}')
        
        plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Soglia 20cm')
        plt.title('Intervento Safety Filter: Scostamento dalla traiettoria ideale')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Deviazione [m]')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        graph_path = "deviazione_sicurezza.png"
        plt.savefig(graph_path)
        print(f"\n Analisi completata. Grafico salvato in: {graph_path}")
    except Exception as e:
        print(f"Errore nella generazione del grafico: {e}")
    # --------------------------------------

    states_array = np.stack(states) if len(states) > 0 else np.zeros((1, sim.n_drones, 6))
    controls_array = np.stack(controls) if len(controls) > 0 else np.zeros((1, sim.n_drones, 6))

    sim_log = {
        "num_drones": sim.n_drones,
        "log_freq": solver_settings.freq,
        "sim_freq": sim.freq,
        "timestamps": np.arange(n_steps) / sim.control_freq,
        # "states": np.array(states),
        # "controls": np.array(controls),
        "states": states_array,
        "controls": controls_array,
        "waypoints": waypoints,
        "simulation_freq": sim.freq,
        "amswarm_every_n_steps": solve_every_n_steps,
        "solve_times": np.array(solve_times),
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
