import inspect

from swarm_gpt.core.motion_primitives import move, rotate, center, swap, spiral_speed, helix, form_circle, zig_zag, form_star, polygon

# Description of helpers injected at runtime into every primitive's scope.
# These are NOT imported by the generated function — they are already available.
ROBOT_API_DESC = """\
import numpy as np  # always available as `np`

def _sanitize_drone_ids(drone_ids: list[int], n_drones: int) -> list[int]:
    # Convert drone IDs from 1-based (Choreographer output) to 0-based (internal indexing).
    # Handles the [...] shorthand (Ellipsis inside a list) as "all drones".
    # ALWAYS use this on the drone_ids param before using them as array indices.
    # Example:
    #   drone_ids, radius = params
    #   drone_ids = np.array(_sanitize_drone_ids(drone_ids, swarm_pos.shape[0]))

def _assign_positions(pos, des_pos):
    # Optimal drone-to-position assignment via Hungarian algorithm (minimises total travel).
    # BOTH pos and des_pos MUST be 2D arrays of shape (n_drones, 3). NEVER pass a 1D array.
    # pos:     NDArray (n_drones, 3) — current drone positions in cm.
    # des_pos: NDArray (n_drones, 3) — ONE desired position per drone (must have n_drones rows).
    # Returns index array: des_pos[result] gives each drone's assigned target in drone order.
    # Typical usage:
    #   assignment = _assign_positions(swarm_pos[drone_ids], des_pos)  # des_pos shape (len(drone_ids), 3)
    #   final_pos  = des_pos[assignment]
    #   waypoints  = {tend: {drone_ids[i]: p.copy() for i, p in enumerate(final_pos)}}

def _form_grid(swarm_pos, limits, height=None, spacing=None):
    # Form a grid of drones at the current position.
    # Returns NDArray (n_drones, 3) in cm, already assigned to drones.
    # spacing: min cm between drones (default 50). height: z override in cm.\
""".strip()

# Source code of three representative existing primitives, loaded at import time.
# GenSwarm equivalent: other_functions_str built from skill_tree.filtered_functions().
FEW_SHOT_EXAMPLES: str = "\n\n".join(
    inspect.getsource(fn) for fn in [move, rotate, center, swap, spiral_speed, helix, form_circle, zig_zag, form_star, polygon]
)
