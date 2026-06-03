#!/usr/bin/env python3
"""Run two Crazyflies with uploaded high-level polynomial trajectories.

This test uses Crazyswarm2's high-level trajectory API:

    uploadTrajectory -> startTrajectory

Unlike cmdFullState streaming, the trajectory is uploaded once and then executed
by the Crazyflie high-level commander.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarm_gpt.core.drone_controller import DroneController  # noqa: E402
from crazyflie_py.uav_trajectory import Polynomial4D, Trajectory  # noqa: E402


TRAJECTORY_ID = 0
GROUP_MASK = 1


def smooth_step_coefficients(distance: float, duration: float) -> np.ndarray:
    """Return coefficients for a 7th-order smooth step from 0 to distance."""
    return np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        35.0 * distance / duration**4,
        -84.0 * distance / duration**5,
        70.0 * distance / duration**6,
        -20.0 * distance / duration**7,
    ], dtype=float)


def zero_coefficients() -> np.ndarray:
    """Return zero polynomial coefficients."""
    return np.zeros(8, dtype=float)


def make_relative_trajectory(*, duration: float, offset: np.ndarray) -> Trajectory:
    """Create a relative high-level trajectory ending at the given xyz offset."""
    trajectory = Trajectory()
    trajectory.polynomials = [
        Polynomial4D(
            duration,
            smooth_step_coefficients(float(offset[0]), duration),
            smooth_step_coefficients(float(offset[1]), duration),
            smooth_step_coefficients(float(offset[2]), duration),
            zero_coefficients(),
        )
    ]
    trajectory.duration = duration
    return trajectory


def motion_offset(
    *,
    drone_index: int,
    initial_position: np.ndarray,
    distance: float,
    pattern: str,
) -> np.ndarray:
    """Return a relative xyz offset for one drone."""
    if pattern == "parallel-x":
        return np.array([distance, 0.0, 0.0], dtype=float)
    if pattern == "split-x":
        direction = 1.0 if drone_index % 2 == 0 else -1.0
        return np.array([direction * distance, 0.0, 0.0], dtype=float)
    if pattern == "outward-y":
        direction = 1.0 if initial_position[1] >= 0.0 else -1.0
        return np.array([0.0, direction * distance, 0.0], dtype=float)
    raise ValueError(f"Unknown pattern: {pattern}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-ids", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--distance", type=float, default=0.5)
    parser.add_argument(
        "--pattern",
        choices=["parallel-x", "split-x", "outward-y"],
        default="parallel-x",
    )
    parser.add_argument("--takeoff-height", type=float, default=1.0)
    parser.add_argument("--takeoff-duration", type=float, default=3.0)
    parser.add_argument("--align-duration", type=float, default=3.0)
    parser.add_argument("--land-duration", type=float, default=4.0)
    parser.add_argument("--timescale", type=float, default=1.0)
    parser.add_argument("--arm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--takeoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--land", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-start", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drone_ids = sorted(set(args.drone_ids))
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.timescale <= 0.0:
        raise ValueError("--timescale must be positive.")
    if args.takeoff_height <= 0.0:
        raise ValueError("--takeoff-height must be positive.")
    if args.land_duration <= 0.0:
        raise ValueError("--land-duration must be positive.")

    controller = DroneController(freq=20.0)
    if not controller._ros_running:
        raise RuntimeError("DroneController did not initialize ROS/Crazyswarm2.")

    available_drones = controller.swarm.allcfs.crazyfliesById
    missing_ids = sorted(set(drone_ids) - set(available_drones))
    if missing_ids:
        raise RuntimeError(f"Drone ids {missing_ids} not found. Available ids: {sorted(available_drones)}")

    selected_drones = {
        drone_id: available_drones[drone_id]
        for drone_id in drone_ids
    }
    controller.swarm.allcfs.crazyfliesById = selected_drones

    if args.arm:
        print(f"Arming drones {drone_ids}")
        controller.arm(True)

    if args.takeoff:
        print(f"Taking off to {args.takeoff_height:.2f} m")
        controller.takeoff(target_height=args.takeoff_height, duration=args.takeoff_duration)

    if args.align_initial:
        print("Moving drones to configured initial x/y positions")
        controller.move_to_initial_positions(height=args.takeoff_height, duration=args.align_duration)

    print("Uploading high-level trajectories")
    for drone_index, (drone_id, crazyflie) in enumerate(selected_drones.items()):
        initial_position = np.array(crazyflie.initialPosition, dtype=float)
        initial_position[2] = args.takeoff_height
        offset = motion_offset(
            drone_index=drone_index,
            initial_position=initial_position,
            distance=args.distance,
            pattern=args.pattern,
        )
        trajectory = make_relative_trajectory(duration=args.duration, offset=offset)
        crazyflie.uploadTrajectory(TRAJECTORY_ID, 0, trajectory)
        print(f"cf{drone_id}: start={initial_position}, end={initial_position + offset}, offset={offset}")

    if args.sync_start:
        print("Starting trajectories with group mask")
        for crazyflie in available_drones.values():
            crazyflie.setGroupMask(0)
        for crazyflie in selected_drones.values():
            crazyflie.setGroupMask(GROUP_MASK)
        controller.swarm.timeHelper.sleep(0.5)
        controller.swarm.allcfs.startTrajectory(
            TRAJECTORY_ID,
            timescale=args.timescale,
            relative=True,
            groupMask=GROUP_MASK,
        )
    else:
        print("Starting trajectories per drone")
        for crazyflie in selected_drones.values():
            crazyflie.startTrajectory(TRAJECTORY_ID, timescale=args.timescale, relative=True)

    flight_time = args.duration * args.timescale
    controller.swarm.timeHelper.sleep(flight_time + 1.0)

    if args.land:
        print(f"Landing selected drones over {args.land_duration:.2f} s")
        for crazyflie in selected_drones.values():
            crazyflie.land(targetHeight=0.06, duration=args.land_duration)
        controller.swarm.timeHelper.sleep(args.land_duration + 1.0)
        for crazyflie in selected_drones.values():
            crazyflie.arm(False)
            crazyflie.setGroupMask(0)

    print("Done")


if __name__ == "__main__":
    main()
