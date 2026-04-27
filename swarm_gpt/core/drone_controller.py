"""Module exposing all necessary functionalities for controlling the real drones."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List

logging._srcfile = None  # Fix logging with rclpy when installed via conda
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import rclpy.time  # noqa: E402
from rclpy.node import Node  # noqa: E402

# The crazyflie_py package does not have a proper installation setup, and crazyswarm2 is potentially
# not installed on the system. Therefore, we need to manually add the crazyswarm2 scripts to the
# Python path. If ROS is not installed, we need to look for the package manually. We bundle these
# steps in the import_utils module.
from swarm_gpt.utils.import_utils import Position, pycrazyswarm  # noqa: E402

if TYPE_CHECKING:
    from scipy.interpolate import BSpline

logger = logging.getLogger(__name__)


class DroneController:
    """Drone controller for the real drones.

    At the core, we use crazyflie_py (crazyswarm2) to control the real drones. The DroneController
    is a wrapper around it that reads in the waypoints and publishes them to the drones.
    """

    def __init__(self, freq: float):
        """Initialize the drone controller.

        Args:
            freq: The frequency at which the controller publishes the drone positions.
        """
        self.freq = freq
        try:
            if not rclpy.ok():
                rclpy.init()
            self._ros_running = True
        except Exception:
            self._ros_running = False
        if not self._ros_running:
            logger.warning("ROS 2 is not running. The drone controller will not be initialized.")
            return
        self._node = Node("swarm_gpt_controller")
        cfg_path = Path(__file__).resolve().parents[3] / "config" / "crazyflies.yaml"
        logger.info("Initializing crazyswarm2")
        self.swarm = pycrazyswarm.Crazyswarm(str(cfg_path))
        self.cmd_pos_pub = {
            id: self._node.create_publisher(Position, f"/cf{id}/cmd_position/", 1)
            for id in self.swarm.allcfs.crazyfliesById.keys()
        }
        self.real_pos_pub = {
            id: self._node.create_publisher(Position, f"/cf{id}/real_position/", 1)
            for id in self.swarm.allcfs.crazyfliesById.keys()
        }

    def requires_ros(fn: Callable) -> Callable:
        """Check if ROS 2 is running before calling the function.

        This function is a decorator that safeguards against calling another function that requires
        ROS 2 if ROS 2 has not been initialized yet. We want to be able to initialize the drone
        controller without ROS 2, e.g., for testing purposes or when running only the simulator.
        However, some functions require ROS 2 to be running, e.g., when publishing the drone
        positions to the ROS 2 topics. If ROS 2 is not running, calls to rclpy might hang
        indefinitely and some attributes of the drone controller might not be initialized.
        Therefore, we protect these functions with this decorator.

        Args:
            fn: The function to decorate.

        Returns:
            The decorated function.
        """

        def requires_ros_wrapper(self: DroneController, *args: Any, **kwargs: Any) -> Any:
            if not self._ros_running:
                raise RuntimeError(f"Function {fn} requires ROS 2, but rclpy is not initialized.")
            return fn(self, *args, **kwargs)

        return requires_ros_wrapper

    @property
    @requires_ros
    def num_drones(self) -> int:
        """Get the number of drones."""
        return len(self.swarm.allcfs.crazyfliesById.keys())

    @requires_ros
    def takeoff(self, target_height: float = 1.0, duration: float = 2.0):
        """Takeoff on all drones.

        Args:
            target_height: The target height to takeoff to.
            duration: The duration of the takeoff.
        """
        self.swarm.allcfs.takeoff(targetHeight=target_height, duration=duration)
        self.swarm.timeHelper.sleep(duration)

    @requires_ros
    def takeoff_low_level(self, target_height: float, duration: float):
        """Takeoff on all drones.

        Args:
            target_height: The target height.
            duration: The duration of the takeoff.
        """
        rate = self._node.create_rate(self.freq)
        drone_pos = {
            drone_id: self.swarm.allcfs.crazyfliesById[drone_id].position()
            for drone_id in self.swarm.allcfs.crazyfliesById.keys()
        }

        for tau in np.linspace(0, 1, int(duration * self.freq)):
            for drone_id in self.swarm.allcfs.crazyfliesById.keys():
                cmd_pos = drone_pos[drone_id].copy()
                cmd_pos[2] = 0.5 - 0.5 * np.cos(tau * np.pi) * target_height
                cmd_msg = self._position_msg(f"drone_{drone_id}_cmd", cmd_pos)
                self.cmd_pos_pub[drone_id].publish(cmd_msg)
            rate.sleep()

    @requires_ros
    def land(self, landing_height: float = 0.02, duration: float = 2.0):
        """Land on all drones.

        For some reason, the drones do not land properly if we use the crazyflie_py land function.
        Therefore, we manually interpolate between the current height and the landing height and
        send the commands to the drones. Afterwards, we call the landing script from crazyflie_py
        to make sure the drones are properly disarmed.

        Args:
            landing_height: The height to land at.
            duration: The duration of the landing.
        """
        rate = self._node.create_rate(self.freq)
        drone_pos = {
            drone_id: self.swarm.allcfs.crazyfliesById[drone_id].position()
            for drone_id in self.swarm.allcfs.crazyfliesById.keys()
        }

        for tau in np.linspace(0, 1, int(duration * self.freq)):
            for drone_id in self.swarm.allcfs.crazyfliesById.keys():
                cmd_pos = drone_pos[drone_id].copy()
                cmd_pos[2] = (1 - tau) * drone_pos[drone_id][2] + tau * landing_height
                cmd_msg = self._position_msg(f"drone_{drone_id}_cmd", cmd_pos)
                self.cmd_pos_pub[drone_id].publish(cmd_msg)
            rate.sleep()
        self.swarm.allcfs.land(targetHeight=landing_height, duration=0.1)
        self.swarm.timeHelper.sleep(0.1)

    def cmd_state(
        self, crazyflie: pycrazyswarm.crazyflie.Crazyflie, pos_ref: np.ndarray, vel_ref: np.ndarray
    ):
        """Send a single control input to the drone.

        Args:
            crazyflie: The crazyflie object.
            pos_ref: The position reference as a numpy array [x, y, z].
            vel_ref: The velocity reference as a numpy array [vx, vy, vz].
        """
        crazyflie.cmdFullState(pos_ref, vel_ref, [0, 0, 0], 0, [0, 0, 0])

    def cmd_position(self, crazyflie: pycrazyswarm.crazyflie.Crazyflie, pos_ref: np.ndarray):
        """Send a single control input to the drone.

        Args:
            crazyflie: The crazyflie object.
            pos_ref: The position reference as a numpy array [x, y, z].
        """
        crazyflie.cmdPosition(pos_ref, yaw=0.0)

    def drone_pose(
        self, crazyflie: pycrazyswarm.crazyflie.Crazyflie
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measure the current pose of the drone.

        Args:
            crazyflie: The crazyflie object.

        Returns:
            The current drone pose as [x y z qx qy qz qw].
        """
        t = crazyflie.tf.lookup_transform("/world", f"/cf{crazyflie.id}", rclpy.time.Time())
        tr = t.transform.translation
        rot = t.transform.rotation
        pose = np.array([tr.x, tr.y, tr.z, rot.x, rot.y, rot.z, rot.w])
        assert pose.shape == (7,), "Pose must have shape (7,)"
        return pose

    def drone_tf_time(self, crazyflie: pycrazyswarm.crazyflie.Crazyflie) -> float:
        """Get the time of the last transform lookup.

        Args:
            crazyflie: The crazyflie object.

        Returns:
            The time of the last transform lookup in nanoseconds.
        """
        t = crazyflie.tf.lookup_transform("/world", f"/cf{crazyflie.id}", rclpy.time.Time())
        stamp = t.header.stamp
        return stamp.sec * 10**9 + stamp.nanosec

    @requires_ros
    def run_open_loop(self, control_inputs: list) -> list:
        """Run open loop control on the real drones.

        Args:
            control_inputs: A list with each element being a dict with drone IDs as keys and array
                of control inputs as values. A control input consists of [x, y, z, vx, vy, vz]
        """
        rate = self._node.create_rate(self.freq)
        drones = self.swarm.allcfs.crazyfliesById
        drone_ids = set(drones.keys())

        pose_data = []

        for control_input in control_inputs:
            pose_data.append({id: self.drone_pose(drones[id]) for id in drone_ids})
            pose_data[-1]["time"] = [self.drone_tf_time(drones[id]) for id in drone_ids]
            for id in drone_ids:
                assert len(control_input[id]) == 6, "Control input must have length 6."
                self.cmd_state(drones[id], control_input[id][0:3], control_input[id][3:6])
            rate.sleep()
            if not rclpy.ok():
                break
        return pose_data

    @requires_ros
    def run_spline_trajectories(self, splines: dict[int, list[BSpline]], duration: float):
        """Run spline controls on the real drones.

        Args:
            splines: A dictionary with drone IDs as keys and lists of B-splines as values.
            duration: The duration of the trajectory.
        """
        rate = self._node.create_rate(self.freq)
        drones = self.swarm.allcfs.crazyfliesById
        drone_ids = set(drones.keys())
        vel_splines = {i: [s.derivative() for s in splines[i]] for i in drone_ids}

        tstart = time.perf_counter()

        while time.perf_counter() - tstart < duration:
            for drone_id in drone_ids:
                t = time.perf_counter() - tstart
                pos = np.array([s(t) for s in splines[drone_id]])
                vel = np.array([s(t) for s in vel_splines[drone_id]])
                self.cmd_state(drones[drone_id], pos, vel)
            rate.sleep()
            if not rclpy.ok():
                break

    @requires_ros
    def _position_msg(self, frame_id: str, position: List[float]) -> Position:
        """Create a Position message.

        Args:
            frame_id: The frame id.
            position: The xyz position.

        Returns:
            The Position message.
        """
        msg = Position()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.x, msg.y, msg.z = position[0], position[1], position[2]
        msg.yaw = 0.0
        return msg
