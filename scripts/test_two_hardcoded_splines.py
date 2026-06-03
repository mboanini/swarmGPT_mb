#!/usr/bin/env python3
"""Run a synchronized hard-coded spline test for two Crazyflies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarm_gpt.core.drone_controller import DroneController  # noqa: E402


def make_offset_splines(
    *,
    duration: float,
    start_position: np.ndarray,
    offset: np.ndarray,
) -> list:
    """Build x/y/z splines from start_position to start_position + offset."""
    times = np.linspace(0.0, duration, 5)
    positions = start_position[None, :] + np.linspace(0.0, 1.0, len(times))[:, None] * offset[None, :]

    return [
        make_interp_spline(times, positions[:, axis_index], k=3)
        for axis_index in range(3)
    ]


def motion_offset(
    *,
    drone_index: int,
    start_position: np.ndarray,
    distance: float,
    pattern: str,
) -> np.ndarray:
    """Return a safe-ish test offset for one drone."""
    if pattern == "parallel-x":
        return np.array([distance, 0.0, 0.0], dtype=float)
    if pattern == "split-x":
        direction = 1.0 if drone_index % 2 == 0 else -1.0
        return np.array([direction * distance, 0.0, 0.0], dtype=float)
    if pattern == "outward-y":
        direction = 1.0 if start_position[1] >= 0.0 else -1.0
        return np.array([0.0, direction * distance, 0.0], dtype=float)
    raise ValueError(f"Unknown pattern: {pattern}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-ids", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--freq", type=float, default=20.0)
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
    parser.add_argument("--land-duration", type=float, default=6.0)
    parser.add_argument("--arm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--takeoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--land", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drone_ids = sorted(set(args.drone_ids))

    controller = DroneController(args.freq)
    if not controller._ros_running:
        raise RuntimeError("DroneController did not initialize ROS/Crazyswarm2.")

    available_drones = controller.swarm.allcfs.crazyfliesById
    missing_ids = sorted(set(drone_ids) - set(available_drones))
    if missing_ids:
        raise RuntimeError(f"Drone ids {missing_ids} not found. Available ids: {sorted(available_drones)}")

    controller.swarm.allcfs.crazyfliesById = {
        drone_id: available_drones[drone_id]
        for drone_id in drone_ids
    }

    if args.arm:
        print(f"Arming drones {drone_ids}")
        controller.arm(True)

    if args.takeoff:
        print(f"Taking off to {args.takeoff_height:.2f} m")
        controller.takeoff(target_height=args.takeoff_height, duration=args.takeoff_duration)

    if args.align_initial:
        print("Moving drones to configured initial x/y positions")
        controller.move_to_initial_positions(height=args.takeoff_height, duration=args.align_duration)

    splines = {}
    for drone_index, drone_id in enumerate(drone_ids):
        start_position = np.array(available_drones[drone_id].initialPosition, dtype=float)
        start_position[2] = args.takeoff_height

        offset = motion_offset(
            drone_index=drone_index,
            start_position=start_position,
            distance=args.distance,
            pattern=args.pattern,
        )
        splines[drone_id] = make_offset_splines(
            duration=args.duration,
            start_position=start_position,
            offset=offset,
        )
        print(f"cf{drone_id}: start={start_position}, end={start_position + offset}")

    try:
        print(
            f"Running synchronized two-drone spline for {args.duration:.2f} s, "
            f"pattern={args.pattern}, distance={args.distance:.2f} m"
        )
        controller.run_spline_trajectories(splines, duration=args.duration)
    finally:
        if args.land:
            print(f"Landing with full-state commands over {args.land_duration:.2f} s")
            controller.land_full_state(landing_height=0.05, duration=args.land_duration)

    print("Done")


if __name__ == "__main__":
    main()
