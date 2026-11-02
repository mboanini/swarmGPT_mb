import inspect

from swarm_gpt.core.motion_primitives import move, rotate, center, swap, spiral_speed, helix, form_circle, zig_zag, form_star, polygon

# Description of helpers injected at runtime into every primitive's scope.
ROBOT_API_DESC = """\

def _sanitize_drone_ids(drone_ids: list[int], n_drones: int) -> list[int]:
    # Convert drone IDs from 1-based (Choreographer output) to 0-based (internal indexing).
    # Handles the [...] shorthand (Ellipsis inside a list) as "all drones".
    # ALWAYS use this on the drone_ids param before using them as array indices.
    # Example:
    #   drone_ids, radius = params
    #   drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])

   
def _assign_positions(pos, des_pos):
    # Optimal drone-to-position assignment via Hungarian algorithm (minimises total travel).
    # BOTH pos and des_pos MUST be 2D arrays of shape (N, 3). NEVER pass a 1D array.
    # N is the number of drones being assigned, either all drones or a selected subset.
    # pos:     NDArray (N, 3) — current positions of the selected drones in cm.
    # des_pos: NDArray (N, 3) — exactly ONE desired position per selected drone.
    # Returns an index array: des_pos[assignment] gives assigned targets
    # in the same drone order as pos.
    # Typical usage for a subset (drone_ids already sanitized to 0-based):
    #   assignment = _assign_positions(swarm_pos[drone_ids], des_pos)
    #   target = des_pos[assignment]
    #   final_pos = swarm_pos.copy()
    #   final_pos[drone_ids] = target
    #   waypoints = {tend: {drone_ids[i]: p.copy() for i, p in enumerate(target)}}


def _form_grid(swarm_pos, limits, height=None, spacing=None):
    # Form a grid of drones at the current position.
    # Returns NDArray (n_drones, 3) in cm, already assigned to drones.
    # spacing: min cm between drones (default 50). height: z override in cm.

def _formation_arrival_time(target_pos, current_pos, tstart, tend) -> float:
    # Physics floor for a "move-then-hold" primitive: earliest feasible arrival time
    # given the bottleneck drone's travel distance.
    # target_pos:  NDArray (N, 3) — assigned target positions in cm.
    # current_pos: NDArray (N, 3) — current positions of the SAME drones, in the SAME order.
    # Returns an absolute arrival timestamp in [tstart, tend], NOT a duration.
    # You usually don't call this directly — `_formation_waypoints` already uses it.

def _formation_waypoints(target_pos, current_pos, tstart, tend, time_to_finish_s, drone_ids=None):
    # Build waypoints for a move-then-hold formation.
    # target_pos: NDArray (N, 3) — assigned target positions in cm.
    # current_pos: NDArray (N, 3) — current positions of the same drones, same order.
    # time_to_finish_s: requested arrival duration, clamped to the physical minimum
    # and to the available interval.
    # drone_ids: global 0-based indices matching the rows of target_pos;
    #            None means indices 0 through N-1.
    # Returns: dict[float, dict[int, NDArray]].
    # Emits the arrival waypoint and subsequent hold waypoints through tend.
    # Waypoint keys are global 0-based drone IDs, including when only a subset moves.
    
""".strip()

# Source code of representative existing primitives - loaded at import time
FEW_SHOT_EXAMPLES: str = "\n\n".join(
    inspect.getsource(fn) for fn in [move, rotate, center, swap, spiral_speed, helix, form_circle, zig_zag, form_star, polygon]
)
