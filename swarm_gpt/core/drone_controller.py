"""Module exposing all necessary functionalities for controlling the real drones."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, List

logging._srcfile = None  # Fix logging with rclpy when installed via conda
import numpy as np  # noqa: E402
# rospy -> rclpy
import rclpy  # noqa: E402
import rclpy.time  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
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
        self._ros_running = False

        # Initialize rclpy so we can check if crazyflie_server is running
        # if rclpy.ok() + check if crazyflie_server in exec
        try:
            if not rclpy.ok():
                rclpy.init()
        except Exception:
            logger.warning("ROS 2 is not running. The drone controller will not be initialized.")
            return

        # Non-blocking check: skip Crazyswarm init if crazyflie_server is not up
        # (Crazyswarm blocks indefinitely on wait_for_service() if the server is absent)
        check_node = rclpy.create_node("swarmgpt_init_check")
        try:
            running_nodes = check_node.get_node_names()
        except Exception:
            running_nodes = []
        finally:
            check_node.destroy_node()

        # if "crazyflie_server" not in running_nodes:
        #     logger.warning("crazyflie_server not running. The drone controller will not be initialized.")
        #     return

        # crazyflie_server is up — Crazyswarm calls rclpy.init() internally,
        # patch it to a no-op since we already initialized above
        _orig_init = rclpy.init
        rclpy.init = lambda *a, **kw: None
        try:
            logger.info("Initializing crazyswarm2")
            self.swarm = pycrazyswarm.Crazyswarm()
            self._ros_running = True
        except Exception as e:
            logger.warning(f"Failed to initialize crazyswarm2: {e}")
        finally:
            rclpy.init = _orig_init

        if not self._ros_running:
            return


        self._node = Node("swarm_gpt_controller")
        # self.cmd_pos_pub = {
        #     # rospy.Publisher(...) -> self._node.create_publisher(...)
        #     id: self._node.create_publisher(Position, f"/cf{id}/cmd_position", 1)
        #     for id in self.swarm.allcfs.crazyfliesById.keys()
        # }
        crazyflies = self.swarm.allcfs.crazyfliesById

        self.cmd_pos_pub = {
            drone_id: self._node.create_publisher(Position, f"{cf.prefix}/cmd_position", 1)
            for drone_id, cf in crazyflies.items()
        }


        self.real_pos_pub = {
            id: self._node.create_publisher(Position, f"/cf{id}/real_position", 1)
            for id in self.swarm.allcfs.crazyfliesById.keys()
        }
        self._drone_pos: dict[int, np.ndarray] = {}
        self._last_cmd_pos: dict[int, np.ndarray] = {}
        # for drone_id in self.swarm.allcfs.crazyfliesById.keys():
        #     cf_name = self.swarm.allcfs.crazyfliesById[drone_id].name
        #     self._node.create_subscription(
        #         PoseStamped,
        #         f"/{cf_name}/pose",
        #         lambda msg, did=drone_id: self._drone_pos.__setitem__(
        #             did, np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        #         ),
        #         1,
        #     )
        for drone_id, cf in self.swarm.allcfs.crazyfliesById.items():
            self._node.create_subscription(
                PoseStamped,
                f"{cf.prefix}/pose",  # cf.prefix already includes leading "/"
                lambda msg, did=drone_id: self._drone_pos.__setitem__(
                    did, np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
                ),
                1,
            )


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
    def arm(self, arm: bool = True, sleep_duration: float = 1.0):
        """Arm or disarm all selected drones.

        Args:
            arm: Whether to arm or disarm the drones.
            sleep_duration: Time to wait after each arm command.
        """
        for crazyflie in self.swarm.allcfs.crazyfliesById.values():
            crazyflie.arm(arm)
            self.swarm.timeHelper.sleep(sleep_duration)

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
    def move_to_initial_positions(self, height: float = 1.0, duration: float = 3.0):
        """Move selected drones to their configured initial x/y positions."""
        for crazyflie in self.swarm.allcfs.crazyfliesById.values():
            goal = np.array(crazyflie.initialPosition, dtype=float)
            goal[2] = height
            crazyflie.goTo(goal, yaw=0.0, duration=duration)
        self.swarm.timeHelper.sleep(duration)

    @requires_ros
    def takeoff_low_level(self, target_height: float, duration: float):
        """Takeoff on all drones.

        Args:
            target_height: The target height.
            duration: The duration of the takeoff.
        """
        # rospy.Rate(...) -> self._node.create_rate(...)
        rate = self._node.create_rate(self.freq)
        # drone_pos = {
        #     drone_id: self.swarm.allcfs.crazyfliesById[drone_id].position()
        #     for drone_id in self.swarm.allcfs.crazyfliesById.keys()
        # }
        rclpy.spin_once(self._node, timeout_sec=1.0)
        drone_pos = {drone_id: self._drone_pos[drone_id].copy() for drone_id in self.swarm.allcfs.crazyfliesById.keys()}

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
        rclpy.spin_once(self._node, timeout_sec=1.0)
        drone_pos = {drone_id: self._drone_pos[drone_id].copy() for drone_id in self.swarm.allcfs.crazyfliesById.keys()}

        for tau in np.linspace(0, 1, int(duration * self.freq)):
            for drone_id in self.swarm.allcfs.crazyfliesById.keys():
                cmd_pos = drone_pos[drone_id].copy()
                cmd_pos[2] = (1 - tau) * drone_pos[drone_id][2] + tau * landing_height
                cmd_msg = self._position_msg(f"drone_{drone_id}_cmd", cmd_pos)
                self.cmd_pos_pub[drone_id].publish(cmd_msg)
            rate.sleep()
        self.swarm.allcfs.land(targetHeight=landing_height, duration=0.1)
        self.swarm.timeHelper.sleep(0.1)

    @requires_ros
    def land_full_state(self, landing_height: float = 0.03, duration: float = 5.0):
        """Land all selected drones with open-loop full-state commands."""
        rclpy.spin_once(self._node, timeout_sec=1.0)
        drones = self.swarm.allcfs.crazyfliesById
        start_pos = {}
        target_pos = {}
        land_vel = {}

        for drone_id, crazyflie in drones.items():
            if drone_id in self._last_cmd_pos:
                position = self._last_cmd_pos[drone_id].copy()
            elif drone_id in self._drone_pos:
                position = self._drone_pos[drone_id].copy()
            elif hasattr(crazyflie, "position"):
                position = np.array(crazyflie.position, dtype=float)
            else:
                position = np.array(crazyflie.initialPosition, dtype=float)

            target = position.copy()
            target[2] = landing_height
            start_pos[drone_id] = position
            target_pos[drone_id] = target
            land_vel[drone_id] = (target - position) / duration

        start_time = self.swarm.timeHelper.time()
        dt = 1.0 / self.freq
        while not self.swarm.timeHelper.isShutdown():
            elapsed = self.swarm.timeHelper.time() - start_time
            if elapsed >= duration:
                break

            tau = elapsed / duration
            for drone_id, crazyflie in drones.items():
                pos = start_pos[drone_id] + tau * (target_pos[drone_id] - start_pos[drone_id])
                self.cmd_state(crazyflie, pos, land_vel[drone_id])
            self.swarm.timeHelper.sleep(dt)

        for drone_id, crazyflie in drones.items():
            self.cmd_state(crazyflie, target_pos[drone_id], np.zeros(3, dtype=float))
            crazyflie.notifySetpointsStop()
            crazyflie.arm(False)

    def cmd_state(
        self, crazyflie: pycrazyswarm.crazyflie.Crazyflie, pos_ref: np.ndarray, vel_ref: np.ndarray
    ):
        """Send a single control input to the drone.

        Args:
            crazyflie: The crazyflie object.
            pos_ref: The position reference as a numpy array [x, y, z].
            vel_ref: The velocity reference as a numpy array [vx, vy, vz].
        """
        crazyflie.cmdFullState(pos_ref, vel_ref, [0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0])

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
        drones = self.swarm.allcfs.crazyfliesById
        drone_ids = set(drones.keys())
        print(len(splines))
        print(splines)
        vel_splines = {i: [s.derivative() for s in splines[i]] for i in drone_ids}

        self.swarm.timeHelper.nextTime = None
        tstart = time.perf_counter()

        while time.perf_counter() - tstart < duration:
            for drone_id in drone_ids:
                t = time.perf_counter() - tstart
                pos = np.array([s(t) for s in splines[drone_id]], dtype=float)
                vel = np.array([s(t) for s in vel_splines[drone_id]], dtype=float)
                self.cmd_state(drones[drone_id], pos, vel)
                self._last_cmd_pos[drone_id] = pos.copy()
            self.swarm.timeHelper.sleepForRate(self.freq)
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
