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
def {function_name}(params, swarm_pos, tstart, tend, limits):
```
- `params`: tuple of N user-supplied values. You decide N, their types and semantics.
- `swarm_pos`: NDArray (n_drones, 3) — current drone positions in cm.
- `tstart`, `tend`: float — time window in seconds. Waypoints must be in (tstart, tend].
- `limits`: dict with keys 'lower' and 'upper', each NDArray[3] in metres.
- Returns `(final_pos, waypoints)`:
  - `final_pos`: NDArray (n_drones, 3) in cm — positions after this primitive ends.
  - `waypoints`: dict[float, dict[int, NDArray]] — timestamp to drone_id to position in cm.

## Available helpers (already in scope — do NOT import them)
```python
{robot_api}
```

## Existing primitives — study these as examples
```python
{other_functions}
```

## Physical constraints
{constraints}

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

## Notes
- The signature is fixed — never add or remove arguments.
- `params` is the only design choice: keep N between 1 and 4.
- The `# n_args: N` comment is mandatory — replace N with the actual count.
- Do NOT write the body — `pass` only.
- Do NOT add imports — `np`, `_assign_positions`, `_form_grid` are already in scope.
- The function name must be exactly `{function_name}`.
""".strip()
