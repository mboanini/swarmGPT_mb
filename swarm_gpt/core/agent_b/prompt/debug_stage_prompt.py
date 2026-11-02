"""Prompt template for the Debug stage (repair of pylint, mechanical and runtime errors).

Produces: the corrected function implementation.

"""

__all__ = ["DEBUG_PROMPT"]

DEBUG_PROMPT = """
The following Python function has errors. Fix them and return the corrected function. Keep the same name, signature and # n_args comment, and unpack `params` with the same names, in the same order and with the same meaning as the Interface.{context_block}

```python
{function_body}
```

Errors:
{error_block}

## Runtime error repair guidance
- Fix the underlying cause of each runtime error, not just the reported waypoint.
- If multiple drones have identical positions, inspect how target positions
  are generated, clipped, and assigned.
- Clipping coordinates independently can collapse distinct positions into
  the same point. Preserve the formation geometry within the arena bounds.
- For continuous trajectories, check whether the parametric equations
  produce identical positions for different drones.
- Preserve consistent drone-to-trajectory assignments across waypoints.
- Do not add ad-hoc collision-avoidance loops or safety filters.

## Helpers already in scope
{robot_api}

{implementation_rules}

## Physical constraints — SAFETY CRITICAL
{constraints}

Return only the corrected function inside a ```python``` block.
""".strip()
