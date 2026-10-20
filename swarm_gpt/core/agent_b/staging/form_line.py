import numpy as np
from numpy.typing import NDArray
from swarm_gpt.core.motion_primitives import _sanitize_drone_ids, _assign_positions, _form_grid, _formation_arrival_time, _formation_waypoints


# n_args: 4
def form_line(params, swarm_pos, tstart, tend, limits):
    '''
    Description: Arranges all drones into a straight line formation along the specified axis in 3D space,
    with adjustable spacing between adjacent drones.

    params: tuple[str, int, int, float] — (axis, spacing, depth, time_to_finish_s)
        axis: str — Specifies the axis ('x', 'y', 'z') along which to form the line.
        spacing: int — Spacing in cm between consecutive drones in the line.
        depth: int — The fixed coordinate in cm for maintaining the line's depth (z when sliding along x and y, or other similar configurations).
        time_to_finish_s: float — Total time allotted to complete the line formation.

    swarm_pos: NDArray (n_drones, 3) — current positions in cm
    tstart: float — start of time window (exclusive)
    tend: float — end of time window (inclusive)
    limits: dict — lower and upper spatial bounds in metres
    return:
        tuple[NDArray, dict[float, dict[int, NDArray]]]
    '''
    axis, spacing, depth, time_to_finish_s = params
    
    n_drones = swarm_pos.shape[0]
    final_pos = swarm_pos.copy()

    # Determine which indices to move along
    axis_indices = {'x': 0, 'y': 1, 'z': 2}
    move_index = axis_indices[axis]

    # Create an array of evenly spaced positions along the selected axis
    line_positions = np.linspace(-200, 200, n_drones) * (2 * (axis == 'x' or axis == 'y') - 1)
    des_pos = np.zeros((n_drones, 3), dtype=float)

    for i in range(n_drones):
        des_pos[i, move_index] = line_positions[i]
        des_pos[i, (move_index + 1) % 3] = depth  # Fixed coordinate
        des_pos[i, (move_index + 2) % 3] = swarm_pos[:, (move_index + 2) % 3].mean()  # Use mean depth

    # Clip the positions to remain within the arena boundaries
    lo_cm, hi_cm = limits['lower'] * 100, limits['upper'] * 100
    np.clip(des_pos, lo_cm, hi_cm, out=des_pos)

    # Assign positions and generate waypoints
    assignment = _assign_positions(swarm_pos, des_pos)
    target = des_pos[assignment]
    waypoints = _formation_waypoints(target, swarm_pos, tstart, tend, time_to_finish_s)
    
    final_pos = swarm_pos.copy()
    final_pos = np.clip(target, lo_cm, hi_cm)  # Ensures all positions are within limits

    return final_pos, waypoints
