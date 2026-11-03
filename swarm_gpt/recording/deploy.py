"""Recording of real-drone deployments: published commands, latest pose/status, events and rosbag.

- Every call of DroneController.cmd_state (the only streamed setpoint path, used by
  run_spline_trajectories and land_full_state) appends one row in memory: wall/monotonic/ROS time,
  Crazyflie ID, flight phase, commanded pos/vel, and the latest /cfX/pose and /cfX/status values
  cached by crazyflie_py. No disk I/O happens inside the control loop; rows are saved when deploy()
  returns or raises.
- `ros2 bag record -a` runs as a separate process for the whole deployment, so all ROS 2 topics
  (/cfX/pose, /cfX/status, /cfX/cmd_full_state, /poses, /tf, /rosout, ...) are stored with their
  original header stamps, independent of the Python process.
"""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_gpt.recording.recorder import REC, save_npz

logger = logging.getLogger(__name__)

COMMAND_FIELDS = {
    "t_wall_ns": "(R,) ns, time.time_ns() when cmd_state was called",
    "t_mono_ns": "(R,) ns, time.perf_counter_ns() (same clock as run_spline_trajectories' t)",
    "t_ros_ns": "(R,) ns, ROS clock of the crazyflie_py node (header stamp of the command)",
    "drone_id": "(R,) Crazyflie ID (from the /cfX prefix)",
    "phase": "(R,) index into meta.phases (deploy step that sent the command)",
    "cmd_pos": "(R,3) m, commanded position (world frame)",
    "cmd_vel": "(R,3) m/s, commanded velocity",
    "spline_t": "(R,) s, exact time at which run_spline_trajectories evaluated the spline for this "
    "command (its t = perf_counter() - tstart); NaN outside run_spline_trajectories",
    "pose_stamp_ns": "(R,) ns, header stamp of the latest /cfX/pose (host receive time), -1 if none",
    "pose_pos": "(R,3) m, latest onboard position estimate (/cfX/pose, 10 Hz by default)",
    "pose_quat": "(R,4) latest onboard orientation estimate [x, y, z, w]",
    "battery_v": "(R,) V, latest /cfX/status battery voltage (1 Hz), NaN if none",
}


class DeployRecorder:
    """Collects one deployment. Created by the AppBackend.deploy hook."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.rows: list[tuple] = []
        self.phases: list[str] = ["outside_phase"]
        self.phase = 0
        self.bag: subprocess.Popen | None = None
        self._bag_log = None
        self.clock_anchors: list[dict] = []
        self.spline_tstart_mono_ns: int | None = None  # run_spline_trajectories' tstart

    def anchor(self, label: str, ros_clock: Any = None) -> None:
        """Sample wall, monotonic and ROS clocks back to back (best of 5 by sampling latency).

        Relates t_mono_ns (used for the spline time) to t_wall_ns and to ROS header stamps
        (rosbag, /cfX/pose) for offline alignment, e.g. with laboratory video.
        """
        try:
            best = None
            for _ in range(5):
                m0 = time.perf_counter_ns()
                wall = time.time_ns()
                ros = ros_clock().now().nanoseconds if ros_clock is not None else None
                m1 = time.perf_counter_ns()
                sample = {"label": label, "t_wall_ns": wall, "t_ros_ns": ros,
                          "t_mono_ns": (m0 + m1) // 2, "sampling_latency_ns": m1 - m0}
                if best is None or sample["sampling_latency_ns"] < best["sampling_latency_ns"]:
                    best = sample
            self.clock_anchors.append(best)
        except Exception:
            logger.exception("DeployRecorder: could not sample clocks")

    # ------------------------------------------------------------------ phases and commands

    def set_phase(self, name: str) -> int:
        previous = self.phase
        if name not in self.phases:
            self.phases.append(name)
        self.phase = self.phases.index(name)
        return previous

    def record_command(self, crazyflie: Any, pos: Any, vel: Any, spline_t: float = np.nan) -> None:
        try:
            t_wall, t_mono = time.time_ns(), time.perf_counter_ns()
            try:
                t_ros = crazyflie.node.get_clock().now().nanoseconds
            except Exception:
                t_ros = -1
            prefix = str(getattr(crazyflie, "prefix", ""))
            drone_id = int(prefix.lstrip("/cf")) if prefix.lstrip("/cf").isdigit() else -1
            stamp, pose_pos, pose_quat = -1, (np.nan,) * 3, (np.nan,) * 4
            pose_stamped = getattr(crazyflie, "poseStamped", None) or {}
            if pose_stamped:
                stamp = pose_stamped["timestamp_sec"] * 10**9 + pose_stamped["timestamp_nsec"]
                p, q = pose_stamped["pose"].position, pose_stamped["pose"].orientation
                pose_pos, pose_quat = (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
            battery = (getattr(crazyflie, "status", None) or {}).get("battery", np.nan)
            self.rows.append((t_wall, t_mono, t_ros, drone_id, self.phase,
                              tuple(np.asarray(pos, dtype=float)),
                              tuple(np.asarray(vel, dtype=float)),
                              stamp, pose_pos, pose_quat, battery, spline_t))
        except Exception:
            logger.exception("DeployRecorder: could not record command")

    def save(self, meta: dict) -> None:
        try:
            if self.rows:
                cols = list(zip(*self.rows))
                save_npz(self.directory / "hardware_commands.npz",
                         t_wall_ns=np.asarray(cols[0], dtype=np.int64),
                         t_mono_ns=np.asarray(cols[1], dtype=np.int64),
                         t_ros_ns=np.asarray(cols[2], dtype=np.int64),
                         drone_id=np.asarray(cols[3]), phase=np.asarray(cols[4]),
                         cmd_pos=np.asarray(cols[5]), cmd_vel=np.asarray(cols[6]),
                         pose_stamp_ns=np.asarray(cols[7], dtype=np.int64),
                         pose_pos=np.asarray(cols[8]), pose_quat=np.asarray(cols[9]),
                         battery_v=np.asarray(cols[10], dtype=float),
                         spline_t=np.asarray(cols[11], dtype=float))
            REC.save_json(self.directory / "meta.json",
                          {**meta, "phases": self.phases, "n_commands": len(self.rows),
                           "clock_anchors": self.clock_anchors,
                           "spline_tstart_mono_ns": self.spline_tstart_mono_ns,
                           "fields": {"hardware_commands.npz": COMMAND_FIELDS}})
        except Exception:
            logger.exception("DeployRecorder: could not save")

    # ------------------------------------------------------------------ rosbag

    def start_rosbag(self) -> None:
        ros2 = shutil.which("ros2")
        if ros2 is None:
            REC.event("rosbag_unavailable", reason="ros2 executable not found")
            return
        try:
            self._bag_log = open(self.directory / "rosbag_record.log", "w")
            self.bag = subprocess.Popen(
                [ros2, "bag", "record", "-a", "-o", str(self.directory / "rosbag")],
                stdout=self._bag_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            REC.event("rosbag_started", pid=self.bag.pid, path=str(self.directory / "rosbag"))
        except Exception as e:
            REC.event("rosbag_error", error=str(e))

    def stop_rosbag(self) -> None:
        if self.bag is None:
            return
        try:
            if self.bag.poll() is None:
                self.bag.send_signal(signal.SIGINT)  # lets rosbag2 close the storage cleanly
                try:
                    self.bag.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.bag.terminate()
                    self.bag.wait(timeout=5)
            REC.event("rosbag_stopped", returncode=self.bag.returncode)
        except Exception as e:
            REC.event("rosbag_error", error=str(e))
        finally:
            if self._bag_log is not None:
                self._bag_log.close()
