"""Motion primitive library."""

import sys
from types import EllipsisType
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

from swarm_gpt.exception import LLMFormatError

motion_primitives = {
    "move": {"n_args": 4},
    "rotate": {"n_args": 2},
    "center": {"n_args": 1},
    "swap": {"n_args": 2},
    "move_z": {"n_args": 2},
    "spiral": {"n_args": 2},
    "spiral_speed": {"n_args": 4},
    "helix": {"n_args": 3},
    "plan": {"n_args": 1},
    "form_circle": {"n_args": 2},
    "zig_zag": {"n_args": 3},
    "wave": {"n_args": 5},
    "twister": {"n_args": 3},
    "form_star": {"n_args": 3},
    "form_cone": {"n_args": 3},
    "polygon": {"n_args": 2},
    # GenSwarm-derived behaviours
    "bridge": {"n_args": 5},
    "encircle": {"n_args": 4},
    "aggregate": {"n_args": 3},
    "cover": {"n_args": 1},
    "flock": {"n_args": 3},
    "explore": {"n_args": 2},
    "pursue": {"n_args": 4},
    "cluster": {"n_args": 2},
    "crossing": {"n_args": 2},
    "shaping": {"n_args": 2},
}


def primitive_by_name(
    name: str,
) -> Callable[
    [tuple, NDArray, float, float, dict[str, NDArray]],
    tuple[NDArray, dict[float, dict[int, NDArray]]],
]:
    """Return a motion primitive by its name."""
    if name not in motion_primitives:
        raise KeyError(f"Unknown motion primitive {name}")
    return getattr(sys.modules[__name__], name)


def rotate(
    params: tuple[int, str],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Rotate all drones by angle theta."""
    angle, axis = params
    angle = np.deg2rad(float(angle))
    steps = int(tend - tstart)  # Number of steps to rotate
    # override rotation to be around z axis atm
    if "z" in axis:
        axis = np.array([0, 0, 1])
    elif "y" in axis:
        axis = np.array([0, 1, 0])
    elif "x" in axis:
        axis = np.array([1, 0, 0])
    else:
        raise LLMFormatError("Invalid axis for rotation")
    max_radius = np.max(np.linalg.norm(swarm_pos[..., :2], axis=-1))
    vmax = 1.0  # Maximum velocity in m/s
    max_angle = (vmax * 100) / max_radius * (tend - tstart)
    angle = np.clip(angle, -max_angle, max_angle)
    r = R.identity() if steps == 0 else R.from_rotvec(axis * angle / steps)

    # Apply the rotation to the vector
    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        swarm_pos = r.apply(swarm_pos)
        waypoints[t] = {i: p.copy() for i, p in enumerate(swarm_pos)}
    return swarm_pos, waypoints


def spiral(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Spiral primitive."""
    n_drones = swarm_pos.shape[0]
    steps, height = params
    # steps = 4
    min_spacing = 60  # Minimum distance between drones in cm

    # Calculate the circumference needed to place all drones with at least the minimum spacing
    start_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    end_radius = min(2 * start_radius, limits["upper"][0] * 100)
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    # Match start positions to drones
    x = start_radius * np.cos(angles)
    y = start_radius * np.sin(angles)
    # TODO: Vary height over time?
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        radius = start_radius + (end_radius - start_radius) * ((t - tstart) / (tend - tstart))
        # Either full rotation around the circle or max angular velocity with 100cm/s linear
        # velocity hard-coded as drone limit
        rot_rate = min(100 / radius, 2 * np.pi / (tend - tstart))
        angles += rot_rate * dt
        swarm_pos = np.array(
            [radius * np.cos(angles), radius * np.sin(angles), [height] * n_drones]
        ).T[assignment]
        waypoints[t] = {i: p.copy() for i, p in enumerate(swarm_pos)}
    return swarm_pos, waypoints


def spiral_speed(
    params: tuple[int, int, int, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Spiral primitive with speed control."""
    steps, height, degrees, increase = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60  # Minimum distance between drones in cm
    steps = int(tend - tstart)

    # Calculate the circumference needed to place all drones with at least the minimum spacing
    start_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    end_radius = min(increase * start_radius, limits["upper"][0] * 100)
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    # Match start positions to drones
    x = start_radius * np.cos(angles)
    y = start_radius * np.sin(angles)
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        radius = start_radius + (end_radius - start_radius) * ((t - tstart) / (tend - tstart))
        # Either full rotation around the circle or max angular velocity with 100cm/s linear
        # velocity hard-coded as drone limit
        rot_rate = min(100 / radius, np.deg2rad(degrees) / (tend - tstart))
        angles += rot_rate * dt
        des_pos = np.array(
            [radius * np.cos(angles), radius * np.sin(angles), [height] * n_drones]
        ).T[assignment]
        waypoints[t] = {i: p.copy() for i, p in enumerate(des_pos)}

    return des_pos, waypoints


def zig_zag(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Moves drones in a zigzag pattern.

    Params:
        params: [steps, delta, delta_h]
        steps: Number of steps (an integer).
        delta: Horizontal displacement per step (an integer).
        delta_h: Vertical displacement per step (an integer).
    """
    steps, delta, delta_h = params
    delta = abs(delta)  # Ensure delta is positive for displacement
    delta_xy = np.abs(np.array([delta, delta, 0]))
    delta_z = np.array([0, 0, delta_h])

    waypoints = {}
    pos = swarm_pos.copy()
    for i, t in enumerate(np.linspace(tstart, tend, steps + 1)[1:]):
        if i == 0:
            pos = _form_grid(swarm_pos, limits=limits)
            waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}
            continue
        displacement_factor = (-1) ** i  # Alternates between 1 and -1
        pos += displacement_factor * delta_xy + delta_z
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def helix(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Helix primitive.

    Drones rise up and circle around the center at the same time.
    """
    steps, delta_h, height = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60  # Minimum distance between drones in cm
    # Calculate the circumference needed to place all drones with at least the minimum spacing
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    # Match start positions to drones
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, [height] * n_drones]).T
    assignment = _assign_positions(swarm_pos, des_pos)
    vmax = 100  # Maximum velocity in cm/s
    rot_rate = min(vmax / radius, 2 * np.pi / (tend - tstart))
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        z = height + (t - tstart) / (tend - tstart) * delta_h
        z = min(z, limits["upper"][2] * 100)
        angles += rot_rate * dt
        pos = np.array([radius * np.cos(angles), radius * np.sin(angles), [z] * n_drones]).T[
            assignment
        ]
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def wave(
    params: tuple[int, int, list[tuple[float, float]], list[float], list[float]],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Specific wave pattern.

    Args:
        params: [steps, height, µ_pairs, aµ1µ2, bµ1µ2]
        swarm_pos: Current positions of the drones.
        tstart: Start time of the primitive.
        tend: End time of the primitive.
        limits: Spatial limits for the drones.
    """
    steps, height, mu_pairs, a_mu, b_mu = params
    steps = int(steps)
    # TODO: Tune default values
    a = 100.0  # Rectangle length
    b = 100.0  # Rectangle Width
    c = np.pi  # Speed of wave propagation
    a_mu = np.array([[0.0, 0.0, 0.25]])  # Shape: (N, 3)
    b_mu = np.array([[0.0, 0.0, 0.25]])  # Shape: (N, 3)
    mu1_mu2 = np.array([[0.4, 0.4]])  # Shape: (N, 2)
    height = max(height, 150)  # Restrict to 75cm for ground effect avoidance

    # Frequencies dictated by dispersion relation
    omega = c * np.pi * np.sqrt((mu1_mu2[:, 0] ** 2) / a**2 + (mu1_mu2[:, 1] ** 2) / b**2)

    # Arrange all drones in a grid like formation
    grid_time = np.linspace(tstart, tend, steps + 1)[1]
    # First step is to form a grid
    waypoints = {}
    swarm_pos = _form_grid(swarm_pos, limits=limits, height=height, spacing=50)
    waypoints[grid_time] = {i: p.copy() for i, p in enumerate(swarm_pos)}

    start_pos = swarm_pos.copy()
    for t in np.linspace(tstart, tend, steps + 1)[2:]:
        # Calculate all sum terms vectorized
        sin_mu1 = np.sin(mu1_mu2[None, :, 0] / a * np.pi * start_pos[:, [0]])  # (n_drones, N)
        sin_mu2 = np.sin(mu1_mu2[None, :, 1] / b * np.pi * start_pos[:, [1]])  # (n_drones, N)
        sin2_term = sin_mu1 * sin_mu2  # (n_drones, N)
        sin_omega_t = np.sin(omega * t)  # (N, )
        cos_omega_t = np.cos(omega * t)  # (N, )
        u_terms = sin2_term[..., None] * (
            a_mu[None, ...] * sin_omega_t + b_mu[None, ...] * cos_omega_t
        )
        # (n_drones, N, 3)
        u = u_terms.sum(axis=1) * 100  # TODO: Remove the 100 factor for scaling to cm
        swarm_pos = start_pos + u
        waypoints[t] = {i: p.copy() for i, p in enumerate(swarm_pos)}

    return swarm_pos, waypoints


def form_star(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a star shape with the drones with {n_drones}//2 spokes."""
    height, min_spacing, delta_radius = params
    min_spacing = max(min_spacing, 40)
    delta_radius = max(delta_radius, 40)
    n_drones = swarm_pos.shape[0]
    drones_per_circle = n_drones // 2
    height = int(height)

    # Calculate the circumference needed to place all drones with at least the minimum spacing
    radius = min_spacing / (2 * np.sin(np.pi / drones_per_circle))

    radii = [radius, radius + delta_radius]
    angle_offset = [0, 2 * np.pi / drones_per_circle]

    des_pos = None
    for r, offset in zip(radii, angle_offset):
        angles = np.linspace(0, 2 * np.pi, drones_per_circle, endpoint=False) + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        if des_pos is None:
            des_pos = np.array([x, y, [height] * drones_per_circle]).T
        else:
            des_pos = np.vstack([des_pos, np.array([x, y, [height] * drones_per_circle]).T])
    # If odd number of drones, put the drone at the center
    if n_drones != drones_per_circle * 2:
        des_pos = np.vstack([des_pos, np.array([0, 0, height]).T])

    assignment = _assign_positions(swarm_pos, des_pos)

    waypoints = {}
    waypoints[tend] = {i: p.copy() for i, p in enumerate(des_pos[assignment])}
    return des_pos[assignment], waypoints


def form_cone(
    params: tuple[int, int, bool],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a cone with the drones."""
    delta_height, spacing, is_inverted = params
    n_drones = swarm_pos.shape[0]

    # Define limits
    start_height = (limits["lower"][2] if is_inverted else limits["upper"][2]) * 100
    delta_height = delta_height * (1 if is_inverted else -1)

    drones_left = n_drones
    drone_increase_per_layer = 4

    # Place first drone
    radius = 0
    z = start_height
    des_pos = np.array([0, 0, z]).T
    drones_left -= 1

    drones_in_layer = 0
    while drones_left > 0:
        drones_in_layer += drone_increase_per_layer
        z += delta_height
        radius = spacing / (2 * np.sin(np.pi / drones_in_layer))

        drones_left -= drones_in_layer
        if drones_left < 0:
            drones_in_layer = drones_left + drones_in_layer

        angles = np.linspace(0, 2 * np.pi, drones_in_layer, endpoint=False)

        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        des_pos = np.vstack([des_pos, np.array([x, y, [z] * drones_in_layer]).T])

    assignment = _assign_positions(swarm_pos, des_pos)

    waypoints = {}
    waypoints[tend] = {i: p.copy() for i, p in enumerate(des_pos[assignment])}
    return des_pos[assignment], waypoints


def twister(
    params: tuple[int, int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a spinning upside-down cone with drones."""
    steps, omega, z_spacing = params
    n_drones = swarm_pos.shape[0]
    # LLM will output omega that is 10x to avoid decimals. TODO: Change this
    omega = omega / 10
    max_omega = 2
    omega = min(omega, max_omega)  # Restrict angular velocity

    lim_lower, lim_upper = limits["lower"], limits["upper"]
    max_radius = min(np.min(lim_upper[:2] - lim_lower[:2] * 100) / 2, 400)
    min_radius = 30

    z_center = 100 * (lim_lower[2] + (lim_upper[2] - lim_lower[2]) / 2)
    max_height = min(z_center + z_spacing * n_drones / 2, lim_upper[2] * 100)
    min_height = max(z_center - z_spacing * n_drones / 2, lim_lower[2] * 100)

    # Calculate the radius and height for each drone
    radius = np.linspace(min_radius, max_radius, n_drones)
    z = np.linspace(min_height, max_height, n_drones)
    angles = np.linspace(0, 4 * np.pi, n_drones)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, z]).T

    assignment = _assign_positions(swarm_pos, des_pos)
    dt = (tend - tstart) / steps

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        angles += omega * dt
        pos = np.array([radius * np.cos(angles), radius * np.sin(angles), z]).T[assignment]
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def center(
    params: tuple[list[int]],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move all the drones to the center, calculated from current position."""
    drone_ids = _sanitize_drone_ids(params[0], swarm_pos.shape[0])
    n_drones = len(drone_ids)
    centroid = np.mean(swarm_pos, axis=0)
    min_spacing = 60  # Minimum distance between drones in cm
    # Calculate the circumference needed to place all drones with at least the minimum spacing
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    des_pos = np.array([x, y, [centroid[2]] * n_drones]).T
    assignment = _assign_positions(swarm_pos[drone_ids], des_pos)
    waypoints = {}
    waypoints[tend] = {i: p.copy() for i, p in enumerate(des_pos[assignment])}
    pos = swarm_pos.copy()
    pos[drone_ids] = des_pos[assignment]
    return pos, waypoints


def form_circle(
    params: tuple[list[int], int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Position drones around the circumference of a circle at height z with at least {min_spacing} cm apart."""
    drone_ids, z_coord = params
    drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    n_drones = len(drone_ids)
    z_coord = int(z_coord)  # z coordinate in cm
    min_spacing = 80  # Minimum distance between drones in cm
    # Calculate the circumference needed to place all drones with at least the minimum spacing
    # If radius is bigger than the limits, make concentric circles
    radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    lim_upper, lim_lower = limits["upper"], limits["lower"]
    max_diameter = min(lim_upper[0] - lim_lower[0], lim_upper[1] - lim_lower[1])
    max_radius = max_diameter * 100 / 2

    radii = [radius]
    drones_per_circle = [n_drones]
    if radius > max_radius:
        n_drones_outer = int(np.pi / np.asin(min_spacing / (2 * max_radius)))
        n_drones_inner = n_drones - n_drones_outer
        radius_outer = max_radius
        radius_inner = min_spacing / (2 * np.sin(np.pi / n_drones_inner))
        radii = [radius_outer, radius_inner]
        drones_per_circle = [n_drones_outer, n_drones_inner]

    des_pos = None
    for r, n in zip(radii, drones_per_circle):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        if des_pos is None:
            des_pos = np.array([x, y, [z_coord] * n]).T
        else:
            des_pos = np.vstack([des_pos, np.array([x, y, [z_coord] * n]).T])

    assignment = _assign_positions(swarm_pos[drone_ids], des_pos)
    waypoints = {}
    waypoints[tend] = {i: p.copy() for i, p in enumerate(des_pos[assignment])}
    pos = swarm_pos.copy()
    pos[drone_ids] = des_pos[assignment]
    return pos, waypoints


def swap(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Swap the positions of two drones."""
    drone1_id, drone2_id = params
    drone1_id, drone2_id = drone1_id - 1, drone2_id - 1
    waypoints = {}
    pos = swarm_pos.copy()
    waypoints[tend] = {drone1_id: pos[drone2_id].copy(), drone2_id: pos[drone1_id].copy()}
    pos[drone1_id], pos[drone2_id] = pos[drone2_id].copy(), pos[drone1_id].copy()
    return pos, waypoints


def move_z(
    params: tuple[list[int], int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move the drones along the z-axis."""
    drone_ids, distance = params
    drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    steps = int(tend - tstart)

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        swarm_pos[drone_ids, 2] = np.clip(swarm_pos[drone_ids, 2] + distance / steps, 100, 200)
        waypoints[t] = {i: swarm_pos[i].copy() for i in drone_ids}

    return swarm_pos, waypoints


def move(
    params: tuple[float, float, float, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Translate move function to waypoints."""
    x, y, z, drone_id = params
    drone_id = drone_id - 1
    swarm_pos[drone_id] = np.array([x, y, z])
    return swarm_pos, {tend: {drone_id: np.array([x, y, z])}}


def _form_grid(
    swarm_pos: NDArray,
    limits: dict[str, NDArray],
    height: float | None = None,
    spacing: int | None = None,
) -> NDArray:
    """Form a grid of drones at the current position."""
    # Get the number of rows and columns
    n_drones = swarm_pos.shape[0]
    rows = int(np.sqrt(n_drones))
    cols = int(np.ceil(n_drones / rows))
    # Get the spacing between the drones
    min_spacing = 50
    spacing = min_spacing if spacing is None else max(spacing, min_spacing)
    x, y = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    lim_upper, lim_lower = limits["upper"], limits["lower"]
    assert (x.max() - x.min()) / 100 <= lim_upper[0] - lim_lower[0], "Grid too wide"
    assert (y.max() - y.min()) / 100 <= lim_upper[1] - lim_lower[1], "Grid too tall"
    x = (x.flatten() - x.mean())[:n_drones]
    y = (y.flatten() - y.mean())[:n_drones]
    centroid = np.mean(swarm_pos, axis=0)
    z = np.full(n_drones, max(10, min(200, centroid[2] if height is None else height)))
    x, y = x + centroid[0], y + centroid[1]
    if (dx := x.max() - lim_upper[0] * 100) > 0:
        x -= dx
    if (dy := y.max() - lim_upper[1] * 100) > 0:
        y -= dy
    if (dx := x.min() - lim_lower[0] * 100) < 0:
        x -= dx
    if (dy := y.min() - lim_lower[1] * 100) < 0:
        y -= dy
    des_pos = np.stack([x, y, z], axis=1)
    assignment = _assign_positions(swarm_pos, des_pos)
    return des_pos[assignment]

def polygon(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a regular polygon with n sides at given height."""
    n_sides, height = params
    height = int(height)
    n_drones = swarm_pos.shape[0]

    # nr drones less than nr sides
    if n_drones < n_sides:
        des_pos = _compute_vertices(n_drones, height, swarm_pos)
    # nr drones more than nr sides 
    elif n_drones > n_sides:
        des_pos = _compute_vertices_and_edges(n_sides, n_drones, height, swarm_pos)
    elif n_drones == n_sides:
        des_pos = _compute_vertices(n_sides, height, swarm_pos)

    assignment = _assign_positions(swarm_pos, des_pos)
    
    waypoints = {}
    waypoints[tend] = {i: p.copy() for i, p in enumerate(des_pos[assignment])}
    return des_pos[assignment], waypoints

    
def _compute_vertices(n_sides: int, height: int, swarm_pos: NDArray):
    """Compute the vertices of a regular polygon and assign drones to them."""
    min_spacing = 60

    # minimum radius
    radius = max(80, int(min_spacing / (2 * np.sin(np.pi / n_sides))) + 10)
    
    cx, cy = np.mean(swarm_pos[:, 0]), np.mean(swarm_pos[:, 1])
    
    # vertices
    angles = [2 * np.pi * i / n_sides + np.pi / 2 for i in range(n_sides)]
    des_pos = np.array([
        [cx + radius * np.cos(a), cy + radius * np.sin(a), height]
        for a in angles
    ])

    return des_pos

def _compute_vertices_and_edges(n_sides, n_drones, height, swarm_pos):
    cx, cy = np.mean(swarm_pos[:, 0]), np.mean(swarm_pos[:, 1])
    radius = max(80, 60 / (2 * np.sin(np.pi / n_sides)))
    
    # Vertici principali
    vertices = np.array([
        [cx + radius * np.cos(2*np.pi*i/n_sides + np.pi/2),
         cy + radius * np.sin(2*np.pi*i/n_sides + np.pi/2),
         height]
        for i in range(n_sides)
    ])
    
    # Droni extra distribuiti sui lati
    extra = n_drones - n_sides
    edge_points = []
    side_idx = 0
    while len(edge_points) < extra:
        v_start = vertices[side_idx % n_sides]
        v_end = vertices[(side_idx + 1) % n_sides]
        # Punto a metà del lato
        midpoint = (v_start + v_end) / 2
        edge_points.append(midpoint)
        side_idx += 1
    
    return np.vstack([vertices, edge_points[:extra]])


def bridge(
    params: tuple[float, float, float, float, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Arrange drones in a straight chain between two XY points at a given height."""
    x1, y1, x2, y2, z = params
    n_drones = swarm_pos.shape[0]
    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    z = np.clip(float(z), lim_lower[2], lim_upper[2])

    t_vals = np.linspace(0, 1, n_drones)
    x = float(x1) + t_vals * (float(x2) - float(x1))
    y = float(y1) + t_vals * (float(y2) - float(y1))
    des_pos = np.stack([x, y, np.full(n_drones, z)], axis=1)

    assignment = _assign_positions(swarm_pos, des_pos)
    waypoints = {tend: {i: p.copy() for i, p in enumerate(des_pos[assignment])}}
    return des_pos[assignment], waypoints


def encircle(
    params: tuple[float, float, float, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Form a ring of drones around a target (target_x, target_y, target_z) with given radius."""
    target_x, target_y, target_z, radius = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60.0
    min_radius = min_spacing / (2 * np.sin(np.pi / n_drones))
    radius = max(float(radius), min_radius)

    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    cx = np.clip(float(target_x), lim_lower[0] + radius, lim_upper[0] - radius)
    cy = np.clip(float(target_y), lim_lower[1] + radius, lim_upper[1] - radius)
    cz = np.clip(float(target_z), lim_lower[2], lim_upper[2])

    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    des_pos = np.stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        np.full(n_drones, cz),
    ], axis=1)

    assignment = _assign_positions(swarm_pos, des_pos)
    waypoints = {tend: {i: p.copy() for i, p in enumerate(des_pos[assignment])}}
    return des_pos[assignment], waypoints


def aggregate(
    params: tuple[float, float, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Gather all drones around a target XY point at a given height."""
    target_x, target_y, z = params
    n_drones = swarm_pos.shape[0]
    min_spacing = 60.0
    radius = min_spacing / (2 * np.sin(np.pi / n_drones)) if n_drones > 1 else 0.0

    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    cx = np.clip(float(target_x), lim_lower[0] + radius, lim_upper[0] - radius)
    cy = np.clip(float(target_y), lim_lower[1] + radius, lim_upper[1] - radius)
    cz = np.clip(float(z), lim_lower[2], lim_upper[2])

    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    des_pos = np.stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        np.full(n_drones, cz),
    ], axis=1)

    assignment = _assign_positions(swarm_pos, des_pos)
    waypoints = {tend: {i: p.copy() for i, p in enumerate(des_pos[assignment])}}
    return des_pos[assignment], waypoints


def cover(
    params: tuple[int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Distribute drones uniformly across the full XY environment at a given height."""
    (z,) = params
    n_drones = swarm_pos.shape[0]
    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    cz = np.clip(float(z), lim_lower[2], lim_upper[2])

    rows = int(np.ceil(np.sqrt(n_drones)))
    cols = int(np.ceil(n_drones / rows))
    xs = np.linspace(lim_lower[0], lim_upper[0], cols)
    ys = np.linspace(lim_lower[1], lim_upper[1], rows)
    xg, yg = np.meshgrid(xs, ys)
    des_pos = np.stack([xg.flatten()[:n_drones], yg.flatten()[:n_drones],
                        np.full(n_drones, cz)], axis=1)

    assignment = _assign_positions(swarm_pos, des_pos)
    waypoints = {tend: {i: p.copy() for i, p in enumerate(des_pos[assignment])}}
    return des_pos[assignment], waypoints


def flock(
    params: tuple[int, float, float],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Simulate boids flocking: cohesion toward the swarm centroid and separation from neighbours.

    Params: steps (int), separation_dist (cm), speed (cm/s).
    """
    steps, separation_dist, speed = params
    steps = max(int(steps), 1)
    separation_dist = max(float(separation_dist), 60.0)
    speed = min(float(speed), 80.0)

    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    dt = (tend - tstart) / steps
    pos = swarm_pos.copy().astype(float)
    waypoints = {}

    # Proportional weights — damped convergence, no oscillation
    cohesion_w = 0.3   # cm/s per cm of distance to centroid
    separation_w = 0.5  # cm/s per cm of overlap

    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        centroid = pos.mean(axis=0)
        vel = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            # Cohesion: proportional pull toward centroid (damps as drones converge)
            v = (centroid[:2] - pos[i, :2]) * cohesion_w

            # Separation: proportional push from each too-close neighbour
            for j in range(pos.shape[0]):
                if i == j:
                    continue
                d_vec = pos[i, :2] - pos[j, :2]
                d = np.linalg.norm(d_vec)
                if 1e-6 < d < separation_dist:
                    v += d_vec / d * (separation_dist - d) * separation_w

            vel[i, :2] = v
            vel[i, 2] = 0.0  # keep altitude stable
            spd = np.linalg.norm(vel[i, :2])
            if spd > speed:
                vel[i, :2] *= speed / spd

        pos = np.clip(pos + vel * dt, lim_lower, lim_upper)
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def explore(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Spread drones to maximise area coverage via mutual XY repulsion.

    Params: steps (int), z (cm).
    """
    steps, z = params
    steps = max(int(steps), 1)
    max_speed = 60.0

    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    cz = np.clip(float(z), lim_lower[2], lim_upper[2])
    dt = (tend - tstart) / steps
    pos = swarm_pos.copy().astype(float)
    pos[:, 2] = cz
    waypoints = {}

    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        vel = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[0]):
                if i == j:
                    continue
                diff = pos[i, :2] - pos[j, :2]
                d = np.linalg.norm(diff)
                if d > 1e-6:
                    vel[i, :2] += diff / (d ** 2) * 5000.0
            # wall repulsion
            for axis in range(2):
                vel[i, axis] += 100.0 / max(pos[i, axis] - lim_lower[axis], 1.0)
                vel[i, axis] -= 100.0 / max(lim_upper[axis] - pos[i, axis], 1.0)
            spd = np.linalg.norm(vel[i, :2])
            if spd > max_speed:
                vel[i, :2] *= max_speed / spd
        pos[:, :2] = np.clip(pos[:, :2] + vel[:, :2] * dt, lim_lower[:2], lim_upper[:2])
        pos[:, 2] = cz
        waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

    return pos, waypoints


def pursue(
    params: tuple[float, float, float, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move the entire swarm formation toward a target point over multiple steps.

    Params: target_x (cm), target_y (cm), target_z (cm), steps (int).
    """
    target_x, target_y, target_z, steps = params
    steps = max(int(steps), 1)
    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    target = np.clip(
        np.array([float(target_x), float(target_y), float(target_z)]),
        lim_lower, lim_upper,
    )

    centroid = swarm_pos.mean(axis=0)
    offsets = swarm_pos - centroid  # preserve formation shape

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        alpha = (t - tstart) / (tend - tstart)
        new_centroid = centroid + alpha * (target - centroid)
        des_pos = np.clip(new_centroid + offsets, lim_lower, lim_upper)
        waypoints[t] = {i: p.copy() for i, p in enumerate(des_pos)}

    final_pos = np.clip(target + offsets, lim_lower, lim_upper)
    return final_pos, waypoints


def cluster(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Divide drones into n_clusters groups evenly spread over the environment.

    Params: n_clusters (int), z (cm).
    """
    n_clusters, z = params
    n_clusters = max(1, int(n_clusters))
    n_drones = swarm_pos.shape[0]
    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    cz = np.clip(float(z), lim_lower[2], lim_upper[2])

    cx_env = (lim_lower[0] + lim_upper[0]) / 2
    cy_env = (lim_lower[1] + lim_upper[1]) / 2
    cluster_radius = min(lim_upper[0] - lim_lower[0], lim_upper[1] - lim_lower[1]) * 0.35

    cluster_angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.column_stack([
        cx_env + cluster_radius * np.cos(cluster_angles),
        cy_env + cluster_radius * np.sin(cluster_angles),
    ])

    groups = np.array_split(np.arange(n_drones), n_clusters)
    des_pos = np.zeros((n_drones, 3))
    intra_radius = 40.0
    for k, members in enumerate(groups):
        n_k = len(members)
        a = np.linspace(0, 2 * np.pi, n_k, endpoint=False)
        r = 0.0 if n_k == 1 else intra_radius / (2 * np.sin(np.pi / n_k))
        des_pos[members, 0] = centers[k, 0] + r * np.cos(a)
        des_pos[members, 1] = centers[k, 1] + r * np.sin(a)
        des_pos[members, 2] = cz

    assignment = _assign_positions(swarm_pos, des_pos)
    waypoints = {tend: {i: p.copy() for i, p in enumerate(des_pos[assignment])}}
    return des_pos[assignment], waypoints


def crossing(
    params: tuple[int, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Two groups of drones swap sides along the X axis over multiple steps.

    Drones are split into left/right halves by their current X position.
    Each half moves to the mirror positions of the other half.
    Params: steps (int), z (cm).
    """
    steps, z = params
    steps = max(int(steps), 1)
    n_drones = swarm_pos.shape[0]
    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    cz = np.clip(float(z), lim_lower[2], lim_upper[2])

    start = swarm_pos.copy().astype(float)
    start[:, 2] = cz

    # Sort drones by X: left half goes to right half's positions and vice versa.
    # Do NOT use Hungarian assignment — it maps each drone to its own reflected
    # position (cost 0) in symmetric formations, causing zero movement.
    sorted_idx = np.argsort(start[:, 0])
    n_left = n_drones // 2
    left_idx = sorted_idx[:n_left]           # indices of leftmost drones
    right_idx = sorted_idx[n_left:]          # indices of rightmost drones (≥ left count)

    des_pos = start.copy()
    # Pair left[i] → right[i]'s start position, right[i] → left[i]'s start position
    for k in range(n_left):
        des_pos[left_idx[k]] = start[right_idx[k]].copy()
        des_pos[right_idx[k]] = start[left_idx[k]].copy()
    # If odd drone count the middle drone (sorted_idx[n_left-1] when n_left==n_right-1)
    # already stays in place — no extra handling needed.

    des_pos[:, :2] = np.clip(des_pos[:, :2], lim_lower[:2], lim_upper[:2])

    waypoints = {}
    for t in np.linspace(tstart, tend, steps + 1)[1:]:
        alpha = (t - tstart) / (tend - tstart)
        interp = start + alpha * (des_pos - start)
        waypoints[t] = {i: p.copy() for i, p in enumerate(interp)}

    return des_pos, waypoints


def shaping(
    params: tuple[list, int],
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Move drones to an arbitrary set of target XY points at a given height.

    Params: target_points (list of [x, y] per drone, in cm), z (cm).
    Uses Hungarian assignment for collision-free allocation.
    """
    target_points, z = params
    n_drones = swarm_pos.shape[0]
    lim_lower, lim_upper = limits["lower"] * 100, limits["upper"] * 100
    cz = np.clip(float(z), lim_lower[2], lim_upper[2])

    pts = np.array([[float(p[0]), float(p[1])] for p in target_points], dtype=float)
    pts = np.clip(pts, lim_lower[:2], lim_upper[:2])

    # Pad or trim to match n_drones
    if len(pts) < n_drones:
        centroid = pts.mean(axis=0)
        pad = np.tile(centroid, (n_drones - len(pts), 1))
        pts = np.vstack([pts, pad])
    pts = pts[:n_drones]

    des_pos = np.column_stack([pts, np.full(n_drones, cz)])
    assignment = _assign_positions(swarm_pos, des_pos)
    waypoints = {tend: {i: p.copy() for i, p in enumerate(des_pos[assignment])}}
    return des_pos[assignment], waypoints


def _sanitize_drone_ids(drone_ids: list[int], n_drones: int) -> list[int]:
    if not isinstance(drone_ids, list):
        raise LLMFormatError(f"Drone IDs must be a list of integers, got {drone_ids}")
    if any(isinstance(i, EllipsisType) for i in drone_ids):
        return list(range(n_drones))
    if not all(isinstance(id, int) for id in drone_ids):
        raise LLMFormatError(f"Drone IDs must be a list of integers, got {drone_ids}")
    return [id - 1 for id in drone_ids]  # TODO: Make LLM assign IDs starting at 0


def _assign_positions(pos: NDArray, des_pos: NDArray) -> NDArray:
    """Assign drones to the closest desired positions.

    Returns:
        The assigned IDs as a numpy array.
    """
    # Get the distance matrix
    dist = np.linalg.norm(pos[:, None, :] - des_pos[None, :, :], axis=-1)
    # Use the Hungarian algorithm to find the optimal assignment
    return linear_sum_assignment(dist)[1]
