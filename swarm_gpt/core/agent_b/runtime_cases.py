"""LLM-generated parameters for each runtime scenario."""
import numpy as np

from swarm_gpt.core.agent_b.infer_test_params import infer as infer_test_params
from swarm_gpt.core.agent_b.parser import param_names
from swarm_gpt.core.agent_b.runtime_check import run as runtime_check
from swarm_gpt.core.agent_b.runtime_scenario import RuntimeScenario


def _scenarios() -> list[RuntimeScenario]:
    limits = {
        "lower": np.array([-1.5, -1.5, 0.25]),
        "upper": np.array([1.5, 1.5, 1.3]),
    }

    scenarios = [
        RuntimeScenario(
            swarm_pos = np.array([
                [-50.0, 0.0, 100.0],
                [50.0, 0.0, 100.0],
            ]),
            tstart=0.0,
            tend=5.0,
            limits=limits,
        ),
        RuntimeScenario(
            swarm_pos=np.array([
                [-50.0, -50.0, 100.0],
                [50.0, -50.0, 100.0],
                [50.0, 50.0, 100.0],
                [-50.0, 50.0, 100.0],
            ]),
            tstart=0.0,
            tend=5.0,
            limits=limits,
        ),
        RuntimeScenario(
            swarm_pos=np.array([
                [-100.0, -50.0, 100.0],
                [0.0, -50.0, 100.0],
                [100.0, -50.0, 100.0],
                [-100.0, 50.0, 100.0],
                [0.0, 50.0, 100.0],
                [100.0, 50.0, 100.0],
            ]),
            tstart=2.0,
            tend=8.0,
            limits=limits,
        ),
    ]

    return scenarios


class TestParamsError(RuntimeError):
    """No valid runtime test inputs could be inferred: an infrastructure failure, not a code bug."""


def validate_test_params(names: list[str], values: tuple, n_drones: int) -> list[str]:
    """Check inferred test inputs against the interface and the scenario's 1-based drone IDs."""
    if len(values) != len(names):
        return [f"expected {len(names)} values {tuple(names)}, got {len(values)}"]
    errors = []
    for name, value in zip(names, values):
        if name == "drone_ids" and value != [...]:  # [...] selects all drones
            if (not isinstance(value, list) or not value or any(type(i) is not int for i in value)
                    or len(set(value)) != len(value) or not all(1 <= i <= n_drones for i in value)):
                errors.append(f"drone_ids must be a non-empty list of distinct ints in "
                              f"1..{n_drones}, got {value!r}")
        elif name.endswith("_id") and (type(value) is not int or not 1 <= value <= n_drones):
            errors.append(f"{name} must be an int in 1..{n_drones}, got {value!r}")
    return errors


def prepare_test_params(function_definition: str, func_name: str) -> list[tuple]:
    """Infer valid parameters for each scenario, once per run.

    An invalid answer is regenerated once with the validation errors as feedback;
    if it is still invalid, TestParamsError is raised.
    """
    names = param_names(function_definition, func_name)
    params = []
    for index, scenario in enumerate(_scenarios()):
        n_drones, feedback = scenario.swarm_pos.shape[0], ""
        for _ in range(2):
            values = infer_test_params(
                function_definition, n_drones=n_drones,
                tstart=scenario.tstart, tend=scenario.tend, case_index=index, feedback=feedback,
            )
            if values is None:
                errors = ["the answer is not a Python tuple"]
            else:
                errors = validate_test_params(names, values, n_drones)
            if not errors:
                break
            feedback = f"Your previous answer {values!r} was rejected: {'; '.join(errors)}"
        else:
            raise TestParamsError(f"Case {index + 1} ({n_drones} drones): {'; '.join(errors)}")
        params.append(values)
    return params


def describe_cases(inferred: list[tuple]) -> str:
    """The validated runtime inputs, for the Debug prompt ('Case k' matches the error prefixes)."""
    lines = [
        f"Case {index + 1}: {scenario.swarm_pos.shape[0]} drones, "
        f"swarm_pos (cm) = {scenario.swarm_pos.tolist()}, tstart = {scenario.tstart}, "
        f"tend = {scenario.tend}, limits (m) = {{'lower': {scenario.limits['lower'].tolist()}, "
        f"'upper': {scenario.limits['upper'].tolist()}}}, "
        f"params = {repr(params).replace('Ellipsis', '...')}"  # show the [...] all-drones notation
        for index, (scenario, params) in enumerate(zip(_scenarios(), inferred))
    ]
    lines.append("These inputs were validated and are valid: support them as given; "
                 "do not filter, clamp or remap drone IDs.")
    return "\n".join(lines)


def check_cases(body: str, func_name: str, inferred: list[tuple]) -> list[str]:
    scenarios = _scenarios()
    if len(inferred) != len(scenarios):
        return ["Runtime certification requires parameters for every scenario."]
    errors = []
    for index, (scenario, params) in enumerate(zip(scenarios, inferred)):
        case_errors = runtime_check(
            body=body, func_name=func_name, test_params=params, scenario=scenario,
        )
        errors.extend(f"Case {index + 1}: {error}" for error in case_errors)
    return errors
