"""LLM-generated parameters for each runtime scenario."""
import numpy as np

from swarm_gpt.core.agent_b.infer_test_params import infer as infer_test_params
from swarm_gpt.core.agent_b.runtime_check import run as runtime_check
from swarm_gpt.core.agent_b.runtime_scenario import RuntimeScenario


def _scenarios() -> list[RuntimeScenario]:
    limits = {
        "lower": np.array([-2.0, -2.0, 0.0]),
        "upper": np.array([2.0, 2.0, 2.0]),
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


def prepare_test_params(function_definition: str, n_args: int) -> list[tuple] | None:
    """Infer all parameters separately for each scenario, once per run."""
    params = []
    for index, scenario in enumerate(_scenarios()):
        values = infer_test_params(
            function_definition,
            n_drones=scenario.swarm_pos.shape[0],
            tstart=scenario.tstart, tend=scenario.tend, case_index=index,
        )
        if values is None or len(values) != n_args:
            return None
        params.append(values)
    return params


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
