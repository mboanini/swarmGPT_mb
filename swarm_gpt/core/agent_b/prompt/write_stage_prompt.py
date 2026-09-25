"""Prompt template for the Write stage (full implementation).

GenSwarm reference: WRITE_LOCAL_FUNCTION_PROMPT_TEMPLATE
Produces: complete function implementation.

Placeholders filled at call time: {function_name}, {function_definition}
Placeholders filled from constants: {task_des}, {env_des}, {robot_api},
                                    {other_functions}, {constraints}
"""

from .constraint_prompt import CONSTRAINTS_TEXT
from .env_description_prompt import ENV_DES
from .robot_api_prompt import FEW_SHOT_EXAMPLES, ROBOT_API_DESC
from .task_description_prompt import TASK_DES

__all__ = ["WRITE_PROMPT", "TASK_DES", "ENV_DES", "ROBOT_API_DESC", "FEW_SHOT_EXAMPLES", "CONSTRAINTS_TEXT"]

WRITE_PROMPT = """
## Background
{task_des}

## Environment
{env_des}

## Role
Implement the body of the motion primitive below.
The signature and docstring have already been designed — preserve them exactly.

## Function to implement
```python
{function_definition}
```

## Timing implementation rules
If `params`:
- Ends with `time_to_finish_s` (target-and-hold shape): use `_formation_waypoints` (and,
  if you need the physical-minimum arrival time it relies on, `_formation_arrival_time`)
  — already in scope, do NOT reimplement their clamping/hold logic. `time_to_finish_s`
  MUST genuinely determine when the arrival waypoint gets written — do NOT just write
  `waypoints[tend] = ...` and ignore it. Study `form_circle`, `form_star`, `form_cone`
  among the existing primitives below.
- Starts with `steps` (continuous-motion shape): compute `dt = (tend - tstart) / steps`
  and generate exactly `steps` waypoints spread across the window. `steps` MUST genuinely
  determine the number of waypoints — do NOT derive an internal step count from
  `tend - tstart` instead (e.g. `int((tend - tstart) / 0.5)`) and ignore the `steps` value
  you were given; a `steps` parameter that doesn't actually drive the waypoint count is a bug.
  Study `spiral`, `helix`, `twister` below for this shape.

## Implementation rules
- **Preserve** `# n_args: N` and the docstring exactly as given above.
- **Single function only** — do NOT define any helper functions or classes inside the code block.
- Do NOT change the function name or signature.
- Do NOT import anything — `np`, `_assign_positions`, `_form_grid`, `_sanitize_drone_ids`,
  `_formation_waypoints`, `_formation_arrival_time` are already in scope.
- Always unpack params with one destructuring line: `drone_ids, p1, p2, ... = params`. Never use index access.
- Call `_sanitize_drone_ids(drone_ids, swarm_pos.shape[0])` before using drone_ids as indices.
- Use `_assign_positions` when assigning drones to new target positions.
- Each drone must occupy a unique position at every timestep - two or more drones cannot be in the same position at the same timestep.
- Waypoint keys are the **0-based** drone indices returned by `_sanitize_drone_ids` (or `range(n_drones)` when all drones move).
- **`final_pos` must be shape `(n_drones, 3)` and cover ALL drones.** If only a subset moves, start from `swarm_pos.copy()` and update only those rows.
- All position arrays must be float64. Initialize from `swarm_pos.copy()` (already float64) or `np.zeros((n, 3), dtype=float)` — never integer dtype.
- Clip all computed positions to the spatial bounds (see above) before putting them in waypoints or `final_pos`.
- Do NOT use while loops, raise, or assert statements.
- Do NOT implement manual collision avoidance (per-step drone separation loops) — the system handles this.
- Never divide by a computed value (e.g. a norm) without first checking it is > 0.
- Always len(waypoints) > 0
- At least one waypoint must be emitted with timestamp strictly in `(tstart, tend]`.
- The final position must be semantically coherent with the intent and the user request.

## CRITICAL — des_pos must always be 2D shape (n_drones, 3)
`_assign_positions(pos, des_pos)` requires BOTH arguments to be 2D `(n_drones, 3)`.
A single point is a 1D array `(3,)` — WRONG. Compute one position per drone.

## Existing primitives — use as implementation reference
```python
{other_functions}
```

## Physical constraints — SAFETY CRITICAL
{constraints}

## Output format
### Reasoning: (plain text only — do NOT include python code blocks here)
### Code:
```python
# n_args: N   <- copy from the definition above, unchanged
def {function_name}(params, swarm_pos, tstart, tend, limits):
    '''
    ... (preserve the docstring from the definition above, unchanged)
    '''
    ... (complete implementation)
```
""".strip()
