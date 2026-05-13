""" 
The control center only performs task allocation at the beginning, and thereafter the robots' autonomous movement is entirely dependent on themselves.
"""

TASK_DES = """
There are Crazyflie 2.1 nano-quadrotor drones operating in a 3D bounded volume.
A centralized orchestrator controls all drones directly — there is NO per-drone autonomous code.

The orchestrator:
- Queries global state using APIs (drone IDs, initial positions, environment bounds, etc.).
- Computes target configurations using any algorithm or numpy logic needed.
- Executes behavior by calling motion/execution APIs (move, form_circle, rotate, polygon, etc.).

Key rules:
- All functions run entirely on the orchestrator.
- Functions EXECUTE behavior (they call APIs and produce movement). They do NOT merely return
  allocation data or position dictionaries for robots to act on later.
- A function should return None unless it computes a sub-result needed by another function
  (e.g., a helper that calculates positions used by the caller to then invoke move()).
- Motion primitives (move, form_circle, rotate, etc.) are blocking calls: they move drones
  and return only when the manoeuvre is complete.
- Query APIs (get_all_drones_id, get_all_drones_initial_position, get_environment_range, etc.)
  return current state and can be called at any time.
- Custom algorithmic logic (numpy, geometry, path planning) must be written inline when
  neither the query APIs nor the motion primitives are sufficient for the task.

Multiple AI assistants collaborate step-by-step to write functions that all run on the
orchestrator. You are one of these assistants.
""".strip()
