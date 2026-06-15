import inspect

from swarm_gpt.core.motion_primitives import bridge, form_circle, rotate

# Description of helpers injected at runtime into every primitive's scope.
# These are NOT imported by the generated function — they are already available.
ROBOT_API_DESC = """\
import numpy as np  # always available as `np`

def _assign_positions(pos, des_pos):
    # Optimal drone-to-position assignment via Hungarian algorithm (minimises total travel).
    # Returns index array: des_pos[result] gives each drone's assigned target in drone order.
    # Typical usage:
    #   assignment = _assign_positions(swarm_pos, des_pos)
    #   final_pos  = des_pos[assignment]
    #   waypoints  = {tend: {i: p.copy() for i, p in enumerate(final_pos)}}

def _form_grid(swarm_pos, limits, height=None, spacing=None):
    # Form a grid of drones at the current position.
    # Returns NDArray (n_drones, 3) in cm, already assigned to drones.
    # spacing: min cm between drones (default 50). height: z override in cm.\
""".strip()

# Source code of three representative existing primitives, loaded at import time.
# GenSwarm equivalent: other_functions_str built from skill_tree.filtered_functions().
FEW_SHOT_EXAMPLES: str = "\n\n".join(
    inspect.getsource(fn) for fn in [form_circle, rotate, bridge]
)