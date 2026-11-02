"""Prompt template for the Write stage (full implementation).

Produces: complete function implementation.

"""

from .constraint_prompt import CONSTRAINTS_TEXT
from .env_description_prompt import ENV_DES
from .implementation_rules_prompt import IMPLEMENTATION_RULES
from .robot_api_prompt import FEW_SHOT_EXAMPLES, ROBOT_API_DESC
from .task_description_prompt import TASK_DES

__all__ = ["WRITE_PROMPT", "IMPLEMENTATION_RULES", "TASK_DES", "ENV_DES", "ROBOT_API_DESC", "FEW_SHOT_EXAMPLES", "CONSTRAINTS_TEXT"]

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
- Ends with `time_to_finish_s` (target-and-hold shape): 
  - use `_formation_waypoints` (and, if you need the physical-minimum arrival time it relies on, `_formation_arrival_time`)
  - already in scope, do NOT reimplement their clamping/hold logic. `time_to_finish_s`
  MUST genuinely determine when the arrival waypoint gets written 
  - do NOT just write `waypoints[tend] = ...` and ignore it. 
  - Study `form_circle`, `form_star`, `form_cone` among the existing primitives below.
- Starts with `steps` (continuous-motion shape): 
  - compute `dt = (tend - tstart) / steps` and generate exactly `steps` waypoints spread across the window. 
  - `steps` MUST genuinely determine the number of waypoints 
  - do NOT derive an internal step count from `tend - tstart` instead (e.g. `int((tend - tstart) / 0.5)`) and ignore the `steps` value
  you were given; 
  - a `steps` parameter that doesn't actually drive the waypoint count is a bug.
  - Study `spiral_speed` and `helix` among the existing primitives below.

{implementation_rules}

## Existing primitives — use as implementation reference
```python
{other_functions}
```

## Available helpers (already in scope - do NOT import or implement them)
```python
{robot_api}
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
