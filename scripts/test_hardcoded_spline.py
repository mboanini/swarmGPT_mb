#!/usr/bin/env python3
"""Run a tiny hard-coded spline through DroneController.
python scripts/test_hardcoded_spline.py --drone-id 6 --distance 0.5 --duration 5.0
python scripts/test_hardcoded_spline.py --drone-id 8 --distance 0.5 --duration 5.0

This is a hardware smoke test for the real-drone path:

    arm -> takeoff -> run_spline_trajectories -> land

Start Crazyswarm2/crazyflie_server first, then run this from the repo root.
Watch the command stream with:

    ros2 topic echo /cf8/cmd_full_state
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from scipy.interpolate import make_interp_spline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarm_gpt.core.drone_controller import DroneController  # noqa: E402


def wait_for_pose(controller: DroneController, drone_id: int, timeout: float) -> np.ndarray | None:
    """Wait for one pose update from DroneController's /cf*/pose subscription."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and rclpy.ok():
        rclpy.spin_once(controller._node, timeout_sec=0.1)
        if drone_id in controller._drone_pos:
            return controller._drone_pos[drone_id].copy()
    return None


def make_test_splines(
    *,
    duration: float,
    base_position: np.ndarray,
    distance: float,
) -> list:
    """Build x/y/z splines for a smooth +x motion."""
    times = np.linspace(0.0, duration, 5)
    x_values = base_position[0] + np.linspace(0.0, distance, len(times))
    y_values = np.full_like(times, base_position[1], dtype=float)
    z_values = np.full_like(times, base_position[2], dtype=float)

    return [
        make_interp_spline(times, x_values, k=3),
        make_interp_spline(times, y_values, k=3),
        make_interp_spline(times, z_values, k=3),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-id", type=int, default=8)
    parser.add_argument("--freq", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--takeoff-height", type=float, default=1.0)
    parser.add_argument("--takeoff-duration", type=float, default=3.0)
    parser.add_argument("--land-duration", type=float, default=5.0)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=-0.5)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--arm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--takeoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--land", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-current-position", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    controller = DroneController(args.freq)
    if not controller._ros_running:
        raise RuntimeError("DroneController did not initialize ROS/Crazyswarm2.")

    drones = controller.swarm.allcfs.crazyfliesById
    if args.drone_id not in drones:
        raise RuntimeError(f"Drone id {args.drone_id} not found. Available ids: {sorted(drones)}")

    controller.swarm.allcfs.crazyfliesById = {args.drone_id: drones[args.drone_id]}

    if args.arm:
        print(f"Arming cf{args.drone_id}")
        controller.arm(True)

    if args.takeoff:
        print(f"Taking off to {args.takeoff_height:.2f} m")
        controller.takeoff(target_height=args.takeoff_height, duration=args.takeoff_duration)

    base_position = np.array([args.x, args.y, args.z], dtype=float)
    if args.use_current_position:
        pose = wait_for_pose(controller, args.drone_id, timeout=3.0)
        if pose is not None:
            base_position = pose
            print(f"Using current pose as spline start: {base_position}")
        else:
            print(f"No pose received; using CLI/default base position: {base_position}")

    distance = args.distance if args.distance is not None else args.amplitude
    if distance is None:
        distance = 1.0

    splines = {args.drone_id: make_test_splines(
        duration=args.duration,
        base_position=base_position,
        distance=distance,
    )}

    try:
        print(f"Running hard-coded +x spline for {args.duration:.2f} s, distance={distance:.2f} m")
        controller.run_spline_trajectories(splines, duration=args.duration)
    finally:
        if args.land:
            print(f"Landing with full-state commands over {args.land_duration:.2f} s")
            controller.land_full_state(landing_height=0.05, duration=args.land_duration)

    print("Done")


if __name__ == "__main__":
    main()
