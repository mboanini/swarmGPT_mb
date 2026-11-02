"""Implementation rules shared by the Write and Debug prompts."""

# Code rules shared by the Write and Debug prompts
IMPLEMENTATION_RULES = """
## Implementation rules
- Preserve `# n_args: N` and the docstring exactly as given above.
- Single function only - do NOT define any helper functions or classes inside the code block.
- Do NOT change the function name or signature.
- Do NOT import anything - `np`, `_assign_positions`, `_form_grid`, `_sanitize_drone_ids`,
  `_formation_waypoints`, `_formation_arrival_time` are already in scope.
- Always unpack params with one destructuring assignment,
  preserving exactly the parameter names and order from the Design docstring.
  Never use index access.
- If drone_ids is declared, sanitize it before indexing.
  If drone_ids is not declared, operate on all drones using swarm_pos.
- Call `_sanitize_drone_ids(drone_ids, swarm_pos.shape[0])` before using drone_ids as indices.
- Use `_assign_positions` when assigning drones to new target positions.
- Every entry in `des_pos` must be unique — no two rows can share the same coordinates. If you build `des_pos` by concatenating multiple sub-arrays (segments, rings, layers…), ensure no coordinate appears in more than one sub-array.
- Waypoint keys are the 0-based drone indices returned by `_sanitize_drone_ids` (or `range(n_drones)` when all drones move).
- `final_pos` must be shape `(n_drones, 3)` and cover ALL drones. If only a subset moves, start from `swarm_pos.copy()` and update only those rows.
- All position arrays must be float64. Initialize from `swarm_pos.copy()` (already float64) or `np.zeros((n, 3), dtype=float)` — never integer dtype.
- Clip all computed positions to the spatial bounds as specified in {env_des} before putting them in waypoints or `final_pos`.
- Do NOT use while loops, raise, or assert statements.
- Generate target positions and waypoints that respect the minimum inter-drone separation specified in the physical constraints,
  including drones that remain stationary.
- Do NOT implement a custom collision-avoidance controller or trajectory safety filter.
- Never divide by a computed value (e.g. a norm) without first checking it is > 0.
- Always len(waypoints) > 0
- At least one waypoint must be emitted with timestamp strictly in `(tstart, tend]`.
- The final position must be semantically coherent with the intent and the user request.

## CRITICAL — des_pos must always be 2D shape (N, 3)
`_assign_positions(pos, des_pos)` requires BOTH arguments to be 2D arrays of shape (N, 3), where N is the number of drones being
assigned (all drones or a selected subset).
A single point is a 1D array `(3,)` — WRONG. 
Compute exactly one target position per selected drone.
""".strip()
