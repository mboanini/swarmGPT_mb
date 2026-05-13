"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

import operator

from modules.deployment.gymnasium_env import (
    GymnasiumTransportationEnvironment,
    GymnasiumHerdingEnvironment,
)
from modules.deployment.gymnasium_env.gymnasium_covering_env import (
    GymnasiumCoveringEnvironment,
)
from run.auto_runner import AutoRunnerBase
from run.utils import evaluate_robot_final_positions


class AutoRunnerCovering(AutoRunnerBase):
    def __init__(
        self,
        env_config_path,
        workspace_path,
        experiment_duration,
        run_mode="rerun",
        target_pkl="WriteRun.pkl",
        script_name="run.py",
        exp_batch=1,
        max_speed=1.0,
        tolerance=0.05,
        test_mode=None,
    ):
        env = GymnasiumCoveringEnvironment(env_config_path)
        super().__init__(
            env_config_path=env_config_path,
            workspace_path=workspace_path,
            experiment_duration=experiment_duration,
            run_mode=run_mode,
            target_pkl=target_pkl,
            script_name=script_name,
            max_speed=max_speed,
            tolerance=tolerance,
            exp_batch=exp_batch,
            env=env,
            test_mode=test_mode,
        )

    def analyze_result(self, run_result) -> dict[str, float]:
        even_metric = evaluate_robot_final_positions(run_result)
        return even_metric

    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        return [
            ("area_ratio", operator.ge, 0.80),
            ("variance_nearest_neighbor_distance", operator.le, 0.1),
        ]
