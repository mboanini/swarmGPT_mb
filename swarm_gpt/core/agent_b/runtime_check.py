"""RuntimeCheck: execute the generated function with test inputs and verify output."""
import numbers
import traceback

import numpy as np

import swarm_gpt.core.motion_primitives as _mp_module
from swarm_gpt.core.agent_b.runtime_scenario import RuntimeScenario

_MIN_DRONE_DISTANCE_CM = 40.0

_TEST_SWARM_POS = np.array([
    [-50., -50., 100.],
    [ 50., -50., 100.],
    [ 50.,  50., 100.],
    [-50.,  50., 100.],
], dtype=float)

_TEST_TSTART = 0.0
_TEST_TEND   = 5.0
_TEST_LIMITS = {
    "lower": np.array([-2., -2.,  0.]),
    "upper": np.array([ 2.,  2.,  2.]),
}

_DEFAULT_SCENARIO = RuntimeScenario(
    swarm_pos = _TEST_SWARM_POS,
    tstart = _TEST_TSTART,
    tend = _TEST_TEND,
    limits = _TEST_LIMITS,
)

def run(body: str, func_name: str, test_params: tuple, scenario: RuntimeScenario | None = None,) -> list[str]:
    """Execute the function with test inputs. Returns a list of error strings (empty = success)."""
    namespace = {
        "np":                    np,
        "_sanitize_drone_ids":   _mp_module._sanitize_drone_ids,
        "_assign_positions":     _mp_module._assign_positions,
        "_form_grid":            _mp_module._form_grid,
        "_formation_arrival_time": _mp_module._formation_arrival_time,
        "_formation_waypoints":    _mp_module._formation_waypoints,
    }

    if scenario is None:
        scenario = _DEFAULT_SCENARIO

    try:
        exec(body, namespace)  # noqa: S102
    except Exception:
        return [f"Exec failed:\n{traceback.format_exc()}"]

    func = namespace.get(func_name)
    if func is None:
        return [f"Function '{func_name}' not found in namespace after exec."]

    try:
        result = func(test_params, scenario.swarm_pos.copy(),scenario.tstart, scenario.tend, {key: value.copy() for key, value in scenario.limits.items()})
    except Exception:
        return [f"Runtime error:\n{traceback.format_exc()}"]

    if not isinstance(result, tuple) or len(result) != 2:
        return [f"Return value must be a tuple (final_pos, waypoints), got {type(result).__name__}"]

    final_pos, waypoints = result
    errors = []

    n_drones = scenario.swarm_pos.shape[0]

    if not isinstance(final_pos, np.ndarray):
        errors.append(f"final_pos must be ndarray, got {type(final_pos).__name__}")
    elif final_pos.shape != (n_drones, 3):
        errors.append(f"final_pos shape must be ({n_drones}, 3), got {final_pos.shape}")
    elif not np.issubdtype(final_pos.dtype, np.number):
        errors.append("final_pos contains invalid numeric values")
    elif np.iscomplexobj(final_pos):
        errors.append("final_pos must contain real coordinates")
    elif not np.all(np.isfinite(final_pos)):
        errors.append("final_pos contains NaN or inf values")

    if not isinstance(waypoints, dict) or len(waypoints) == 0:
        errors.append("waypoints must be a non-empty dict")
    else:
        current_positions = scenario.swarm_pos.copy()
        for t in waypoints:
            if (
                isinstance(t, (bool, np.bool_))
                or not isinstance(t, numbers.Real)
                or not np.isfinite(t)
            ):
                errors.append(
                    f"Invalid waypoint timestamp: {t!r}. "
                    "Timestamp must be a finite real number."
                )
                break
            if not (scenario.tstart < t <= scenario.tend):
                errors.append(
                    f"Waypoint timestamp {t!r} outside primitive interval "
                    f"({scenario.tstart}, {scenario.tend}]."
                )
                break
        if errors:
            return errors

        for t, positions in sorted(waypoints.items()):
            if not isinstance(positions, dict):
                errors.append(f"waypoints[{t}] must be a dict, got {type(positions).__name__}")
                break
            valid_positions = {}
            for drone_id, pos in positions.items():
                if (
                    isinstance(drone_id, (bool, np.bool_))
                    or not isinstance(drone_id, (int, np.integer))
                    or not (0 <= drone_id < n_drones)
                ):
                    errors.append(f"waypoints[{t}]: invalid drone ID {drone_id!r}")
                    break
                if not isinstance(pos, np.ndarray):
                    errors.append(f"waypoints[{t}][{drone_id}] must be ndarray")
                    break
                if pos.shape != (3,):
                    errors.append(
                        f"waypoints[{t}][{drone_id}] must have shape (3,), got {pos.shape}"
                    )
                    break
                if not np.issubdtype(pos.dtype, np.number) or np.iscomplexobj(pos):
                    errors.append(f"waypoints[{t}][{drone_id}] contains invalid numeric values")
                    break
                if not np.all(np.isfinite(pos)):
                    errors.append(f"waypoints[{t}][{drone_id}] contains NaN or inf")
                    break
                valid_positions[drone_id] = pos
            else:
                for drone_id, pos in valid_positions.items():
                    current_positions[drone_id] = pos

                for i in range(n_drones):
                    for j in range(i + 1, n_drones):
                        dist = np.linalg.norm(
                            current_positions[i] - current_positions[j]
                        )

                        if dist < _MIN_DRONE_DISTANCE_CM:
                            errors.append(
                                f"waypoints[{t}]: drones {i} and {j} are too close "
                                f"({dist:.1f} cm < {_MIN_DRONE_DISTANCE_CM} cm)"
                            )
            if errors:
                break
        # Check consistency between final_pos and the last waypoint state.
        if (
            not errors
            and isinstance(final_pos, np.ndarray)
            and final_pos.shape == (n_drones, 3)
            and np.all(np.isfinite(final_pos))
        ):
            for drone_id in range(n_drones):
                if not np.allclose(
                    final_pos[drone_id],
                    current_positions[drone_id],
                    rtol=0,
                    atol=1e-6,
                ):
                    errors.append(
                        f"final_pos[{drone_id}] does not match "
                        f"the last waypoint state: "
                        f"final_pos={final_pos[drone_id]}, "
                        f"waypoint={current_positions[drone_id]}"
                    )
    return errors
