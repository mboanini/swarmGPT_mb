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
Your task is to implement the body of the motion primitive below.
The signature and docstring have already been designed — do not change them.

## Function to implement
```python
{function_definition}
```

## Available helpers (already in scope — do NOT import them)
```python
{robot_api}
```

## Existing primitives — use these as implementation reference
```python
{other_functions}
```

## Physical constraints
{constraints}

## Output format
### Reasoning: (step-by-step implementation strategy)
### Code:
```python
# n_args: N   <- copy from the definition above, unchanged
def {function_name}(params, swarm_pos, tstart, tend, limits):
    '''
    ... (preserve the docstring from the definition above, unchanged)
    '''
    ... (complete implementation)
```

## Notes
- Preserve the `# n_args: N` comment and the docstring exactly as given above.
- Do NOT change the function name or signature.
- Do NOT import anything — `np`, `_assign_positions`, `_form_grid` are already in scope.
- All positions in cm. Multiply `limits` values by 100 to compare with positions.
- Use `_assign_positions` when assigning drones to new target positions.
- Do NOT use while loops, raise, or assert statements.
- At least one waypoint must be emitted with timestamp strictly in (tstart, tend].
- Strictly follow the output format.
""".strip()
