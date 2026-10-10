"""Prompt template for the Design stage (interface-only, no body).

GenSwarm reference: DESIGN_LOCAL_FUNCTION_PROMPT_TEMPLATE
Produces: # n_args: N comment + signature + docstring + pass (no body).

Placeholders filled at call time: {function_name}, {function_des}
Placeholders filled from constants: {task_des}, {env_des}, {robot_api},
                                    {other_functions}, {constraints}
"""

from .constraint_prompt import CONSTRAINTS_TEXT
from .env_description_prompt import ENV_DES
from .robot_api_prompt import FEW_SHOT_EXAMPLES, ROBOT_API_DESC
from .task_description_prompt import TASK_DES

__all__ = ["DESIGN_PROMPT", "TASK_DES", "ENV_DES", "ROBOT_API_DESC", "FEW_SHOT_EXAMPLES", "CONSTRAINTS_TEXT"]

DESIGN_PROMPT = """
## Background
{task_des}

## Environment
{env_des}

## Role
Your task is to design the **interface** of the motion primitive `{function_name}`.
Decide what the `params` tuple contains: how many values (N), their types, their meaning.
Do NOT implement the function body — write `pass` only.

## Behaviour to implement
{function_des}

## Fixed signature — never change this
Every motion primitive has exactly this signature:
```python
# n_args: N
def {function_name}(
    params: tuple,
    swarm_pos: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
```
- `params`: tuple of N user-supplied values. You decide N, their types and semantics.
- `swarm_pos`: NDArray (n_drones, 3) — units: cm, axes: x=right, y=forward, z=up.
- `tstart`, `tend`: float — time window in seconds. Waypoints must be in (tstart, tend].
- `limits`: dict with keys 'lower' and 'upper', each NDArray[3] in metres.
- Returns `(final_pos, waypoints)`:
  - `final_pos`: NDArray (n_drones, 3) in cm — positions after this primitive ends.
  - `waypoints`: dict[float, dict[int, NDArray]] — timestamp to drone_id to position in cm.

## Timing Rule
Pick whichever of these two shapes matches `{function_des}`.
- Reaches a shape/position and then holds still (most formations and simple moves): the
  LAST element of `params` must be a float named `time_to_finish_s`. Study `form_circle`,
  `form_star`, `form_cone` among the existing primitives below for this shape.
- A continuous, evolving motion where the drone keeps moving throughout — a spiral, a wave,
  a spin, a coverage sweep, a patrol path, a systematic traversal — that should genuinely
  take longer the more of it there is: the FIRST element of `params` must be an integer
  named `steps`. Study `spiral`, `helix`, `twister` below for this shape.

Decision test: does the drone keep moving for the full duration, or does it arrive
somewhere and stop? Keeps moving -> `steps`. Arrives and stops -> `time_to_finish_s`.

## Available helpers (already in scope — do NOT import or implement them)
```python
{robot_api}
```

## Existing primitives — study these as examples
```python
{other_functions}
```

## Physical constraints
{constraints}

## Notes
- The signature is fixed — never add or remove arguments.
- `params` is the only design choice: keep N between 1 and 4.
- The `# n_args: N` comment is mandatory — replace N with the TOTAL number of elements in the
  params tuple. If drone_ids is the first element, it counts: e.g. `(drone_ids, radius, height)`
  → N=3, not N=2. N must equal exactly the number of comma-separated items in the params tuple.
- The docstring line `params: tuple[type, ...] — (name1, name2, ...)` is mandatory, on one line:
  list the N parameter names in order, names only (no types), separated by commas.
- Do NOT write the body — `pass` only.
- Do NOT add imports — `np`, `_assign_positions`, `_form_grid` are already in scope.
- The function name must be exactly `{function_name}`.

## Output format
### Reasoning: (what params make sense for this behaviour, and why)
### Code:
```python
# n_args: N
def {function_name}(params, swarm_pos, tstart, tend, limits):
    '''
    Description: ...

    params: tuple[type, ...] — (param1_name, param2_name, ...)
        param1_name: type — what it controls
        param2_name: type — what it controls
    swarm_pos: NDArray (n_drones, 3) — current positions in cm
    tstart: float — start of time window (exclusive)
    tend: float — end of time window (inclusive)
    limits: dict — lower and upper spatial bounds in metres
    return:
        tuple[NDArray, dict[float, dict[int, NDArray]]]
    '''
    pass
```
""".strip()
