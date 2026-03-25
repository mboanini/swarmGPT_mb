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

from swarm_gpt.core import Choreographer, DroneController
from swarm_gpt.core.sim import simulate_axswarm
from swarm_gpt.exception import LLMException

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
import sys, pickle, logging
logging.basicConfig(level=logging.WARNING)

data_path = sys.argv[1]
with open(data_path, "rb") as f:
    payload = pickle.load(f)

splines   = payload["splines"]    # dict drone -> list of scipy splines
settings  = payload["settings"]
duration  = payload["duration"]

from collections import deque
import time
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from swarm_gpt.utils.utils import draw_line

fps          = 60
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
sim.reset()
sim.state_control(__import__("numpy").random.random((1, sim.n_drones, 13)))
sim.step(sim.freq // sim.control_freq)
sim.reset()

vel_splines = {k: [s.derivative() for s in v] for k, v in splines.items()}
pos = np.array([[s(0) for s in splines[j]] for j in splines])[None, ...]
sim.data = sim.data.replace(
    states=sim.data.states.replace(pos=sim.data.states.pos.at[...].set(pos))
)

rng   = np.random.default_rng(0)
rgbas = rng.random((sim.n_drones, 4))
rgbas[..., 3] = 1
swarm_pos = [deque(maxlen=100) for _ in range(sim.n_drones)]

tstart   = time.time()
n_steps  = int(duration * sim.control_freq)

try:
    for i in range(n_steps):
        ct = i / sim.control_freq
        des_pos = np.array([[s(ct) for s in splines[j]] for j in splines])
        des_vel = np.array([[s(ct) for s in vel_splines[j]] for j in splines])
        controls = np.concatenate(
            (des_pos, des_vel, np.zeros((sim.n_drones, 7))), axis=-1
        )[None, ...]
        sim.state_control(controls)
        sim.step(sim.freq // sim.control_freq)

        if ((i * fps) % sim.control_freq) < fps:
            for j, dq in enumerate(swarm_pos):
                dq.append(np.asarray(sim.data.states.pos[0, j]))
                draw_line(sim, np.array(dq), rgba=rgbas[j % len(rgbas)], min_size=2, max_size=5)
            sim.render()
            dt = ct - (time.time() - tstart)
            if dt > 0:
                time.sleep(dt)
finally:
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
        model_id: str = "gpt-4o-2024-05-13",
        use_motion_primitives: bool = False,
    ):
        self.root_path = Path(__file__).resolve().parents[2]
        with open(self.root_path / "swarm_gpt/data/settings.yaml", "r") as f:
            self.settings = yaml.safe_load(f)

        self.waypoints = None
        self.splines: dict = {}
        self.drone_controller = DroneController(20)
        self.choreographer = Choreographer(
            config_file=config_file,
            model_id=model_id,
            use_motion_primitives=use_motion_primitives,
        )
        self.mode: Literal["preset", "real"] = "real"
        self._preset: None | str = None
        self._strict_processing = strict_processing
        self._strict_drone_match = strict_drone_match

        # Subprocess visualizer state
        self._viz_proc: subprocess.Popen | None = None
        self._viz_data_path: str | None = None   # path del file pickle temporaneo

    # ── Subprocess visualizer helpers ─────────────────────────────────────────

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

    def _start_viz(self, splines: dict, duration: float) -> None:
        """Kill any existing visualizer, serialize splines to disk, launch new subprocess."""
        self._stop_viz()

        # Serializza le splines in un file temporaneo (pickle supporta oggetti scipy)
        payload = {
            "splines": {k: list(v) for k, v in splines.items()},
            "settings": dict(self.settings),
            "duration": duration,
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

    # ── Core methods ──────────────────────────────────────────────────────────

    @property
    def presets(self) -> list[str]:
        return [s.name for s in (self.root_path / "swarm_gpt/data/presets").glob("*")]

    @self_correct(n_retries=2)
    def initial_prompt(self, text: str, *, response: str | None = None) -> list[dict[str, str]]:
        logger.info("Generating initial choreography for: %s", text)
        self.choreographer.reset_history()
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
        if hasattr(self, "sim_results"):
            del self.sim_results
        assert self.waypoints is not None, "Please generate a choreography first"

        sim_data = None
        for key, data, total in simulate_axswarm(self.waypoints, self.settings, gui=False):
            if key == "progress":
                yield key, data, total
            else:
                sim_data = data

        assert sim_data is not None, "simulate_axswarm returned no data"

        t = sim_data["timestamps"][::10]
        lam = 0.1
        self.splines.clear()
        for i, drone in self.choreographer.agents.items():
            controls = sim_data["controls"][:, i, :3]
            self.splines[drone] = [
                make_smoothing_spline(t, controls[:, j], lam=lam) for j in range(3)
            ]

        if gui:
            self._start_viz(self.splines, float(t[-1]))

        logger.info("Simulation successful")
        return sim_data

    def replay(self) -> None:
        """Replay the last simulation without re-running axswarm."""
        assert self.splines, "Please run Simulate first before replaying."
        assert self.waypoints is not None, "Waypoints missing."
        duration = float(np.max(self.waypoints["time"][:, -1]))
        logger.info("Starting replay (duration=%.2f s)", duration)
        self._start_viz(self.splines, duration)

    def deploy(self, drone_ids: list[int] | None = None):
        logger.info("Deploying drones")
        assert self.splines, "Please run the simulation first!"
        if not self.drone_controller._ros_running:
            raise RuntimeError("ROS is not running. Please start ROS before deploying.")

        if drone_ids is not None:
            cfs = self.drone_controller.swarm.allcfs.crazyfliesById
            cfs = {k: v for k, v in cfs.items() if k in drone_ids}
            self.drone_controller.swarm.allcfs.crazyfliesById = cfs

        for i, drone in enumerate(self.drone_controller.swarm.allcfs.crazyfliesById.values()):
            drone.setLEDColor(*colors[i % len(colors)])

        original_song = self.music_manager.song
        duration = next(iter(self.waypoints.values()))[-1, 0]
        try:
            self.music_manager.song = original_song + "[deploy]"
        except AssertionError:
            pass
        self.drone_controller.takeoff(target_height=1.0, duration=3.0)
        self.music_manager.play()
        self.drone_controller.run_spline_trajectories(self.splines, duration=duration)
        self.drone_controller.land()
        self.music_manager.song = original_song
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
        for attr in ("full_trajectory", "sim_results"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.choreographer.reset_history()
        logger.info("Backend reset complete.")