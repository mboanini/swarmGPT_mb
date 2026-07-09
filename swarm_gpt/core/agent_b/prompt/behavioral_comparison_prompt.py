from .constraint_prompt import CONSTRAINTS_TEXT
from .env_description_prompt import ENV_DES
from .robot_api_prompt import FEW_SHOT_EXAMPLES, ROBOT_API_DESC
from .task_description_prompt import TASK_DES

__all__ = ["BEHAVIORAL_COMPARISON_PROMPT", "CODE_REVIEW_PROMPT"]

BEHAVIORAL_COMPARISON_PROMPT = """

## Background:
{task_des}

## Task:
Please summarize the main functions and boundary conditions that `{function_name}` should implement,
based on this description:
** Description: ** {function_des}
and the user original instruction:
** User original instruction: ** {user_command}

Then read the code below and describe what functions the code actually completes and how the key steps are implemented:
```python
{function_body}
```

## Available context
```python
{robot_api}
```
```python
{other_functions}
```

Finally, compare the code behavior with the requirements point-by-point to determine whether they are consistent. 

## Output format
Respond with exactly one of:
SATISFIED
NOT_SATISFIED: <ordered list of the specific mismatches found>

Do not propose a fix. Only judge consistency.

""".strip()

CODE_REVIEW_PROMPT = """

## Background:
{task_des}

## Role:
You are fixing a specific, already-diagnosed inconsistency in `{function_name}`.
The implementation was audited against its description and found NOT_SATISFIED for the following specific reasons:

{mismatch_list}

Your job is ONLY to resolve these listed mismatches. Do not look for additional issues beyond what is listed above.

## These are the User original instructions:
{user_command}

## These are the environment description:
There are the basic descriptions of the environment.
{env_des}

## Existing primitives — there are the existing functions that you can use as style and implementation reference:
```python
{other_functions}
```

## Available helpers (already in scope — do NOT import) - these are functions that you can directly call:
```python
{robot_api}
```

## These are the description and the interface of the primitive to review - DO NOT modify these:
**Description:** {function_des}

**Interface (DO NOT modify — signature, # n_args comment, and docstring are fixed):**
```python
{function_definition}
```

## This is the implementation that need to be checked:
```python
{function_body}
```

## These are constraints that the function should satisfy:
{constraints}

## Output format
Reasoning: for each mismatch listed above, point to the exact erroneous
line(s) of code and explain the correction you will make. Write code
directly here without ```python``` fences — those are reserved for the
final function block below.

Modified function:
```python
def {function_name}(params, swarm_pos, tstart, tend, limits):
    '''
    ... (docstring unchanged, content may be updated)
    '''
    ... (corrected body)
```

## Rules
- Fix ONLY the mismatches listed above — nothing else. Do not rewrite
  parts of the function that were not flagged.
- Do NOT change the function name, signature, or `# n_args: N` comment.
- Preserve the docstring; you may update its content if the fix changes
  documented behavior, but do not remove it.
- Do NOT import anything — `np`, `_assign_positions`, `_form_grid`,
  `_sanitize_drone_ids` are already in scope.
- If the velocity is part of the output, it must be normalized.
- If resolving a mismatch requires touching code unrelated to it, explain
  why in the Reasoning section rather than silently expanding the change.
- No ```python``` fences anywhere except inside the Modified function block.
""".strip()