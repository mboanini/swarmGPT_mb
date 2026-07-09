"""RuntimeCheck: execute the generated function with test inputs and verify output."""
import traceback

import numpy as np

import swarm_gpt.core.motion_primitives as _mp_module

_MIN_DRONE_DISTANCE_CM = 60.0

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


def run(body: str, func_name: str, test_params: tuple) -> list[str]:
    """Execute the function with test inputs. Returns a list of error strings (empty = success)."""
    namespace = {
        "np":                    np,
        "_sanitize_drone_ids":   _mp_module._sanitize_drone_ids,
        "_assign_positions":     _mp_module._assign_positions,
        "_form_grid":            _mp_module._form_grid,
    }

    try:
        exec(body, namespace)  # noqa: S102
    except Exception:
        return [f"Exec failed:\n{traceback.format_exc()}"]

    func = namespace.get(func_name)
    if func is None:
        return [f"Function '{func_name}' not found in namespace after exec."]

    try:
        result = func(test_params, _TEST_SWARM_POS.copy(), _TEST_TSTART, _TEST_TEND, _TEST_LIMITS)
    except Exception:
        return [f"Runtime error:\n{traceback.format_exc()}"]

    if not isinstance(result, tuple) or len(result) != 2:
        return [f"Return value must be a tuple (final_pos, waypoints), got {type(result).__name__}"]

    final_pos, waypoints = result
    errors = []

    n_drones = _TEST_SWARM_POS.shape[0]

    if not isinstance(final_pos, np.ndarray):
        errors.append(f"final_pos must be ndarray, got {type(final_pos).__name__}")
    elif final_pos.shape != (n_drones, 3):
        errors.append(f"final_pos shape must be ({n_drones}, 3), got {final_pos.shape}")
    elif not np.all(np.isfinite(final_pos)):
        errors.append("final_pos contains NaN or inf values")

    if not isinstance(waypoints, dict) or len(waypoints) == 0:
        errors.append("waypoints must be a non-empty dict")
    else:
        for t, positions in waypoints.items():
            if not isinstance(positions, dict):
                errors.append(f"waypoints[{t}] must be a dict, got {type(positions).__name__}")
                break
            valid_positions = {}
            for drone_id, pos in positions.items():
                if not isinstance(pos, np.ndarray):
                    errors.append(f"waypoints[{t}][{drone_id}] must be ndarray")
                    break
                if not np.all(np.isfinite(pos)):
                    errors.append(f"waypoints[{t}][{drone_id}] contains NaN or inf")
                    break
                valid_positions[drone_id] = pos
            else:
                ids = list(valid_positions.keys())
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        dist = np.linalg.norm(valid_positions[ids[i]] - valid_positions[ids[j]])
                        if dist < _MIN_DRONE_DISTANCE_CM:
                            errors.append(
                                f"waypoints[{t}]: drones {ids[i]} and {ids[j]} are too close "
                                f"({dist:.1f} cm < {_MIN_DRONE_DISTANCE_CM} cm)"
                            )

    return errors
