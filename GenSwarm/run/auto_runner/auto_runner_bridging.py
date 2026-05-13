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

import numpy as np

from modules.deployment.gymnasium_env import GymnasiumBridgingEnvironment
from run.auto_runner import AutoRunnerBase
from run.utils import evaluate_shape_similarity


class AutoRunnerBridging(AutoRunnerBase):
    def __init__(
        self,
        env_config_path,
        workspace_path,
        experiment_duration,
        run_mode="rerun",
        target_pkl="WriteRun.pkl",
        script_name="run.py",
        exp_batch=1,
        test_mode=None,
        max_speed=1.0,
        tolerance=0.05,
    ):
        self.env = GymnasiumBridgingEnvironment(env_config_path)
        super().__init__(
            env_config_path=env_config_path,
            workspace_path=workspace_path,
            experiment_duration=experiment_duration,
            run_mode=run_mode,
            target_pkl=target_pkl,
            script_name=script_name,
            max_speed=max_speed,
            exp_batch=exp_batch,
            test_mode=test_mode,
            tolerance=tolerance,
            env=self.env,
        )

    def analyze_result(self, run_result) -> dict[str, float]:
        robots_num = self.env.num_robots
        target_y = np.linspace(-2, 2, robots_num)
        target_x = np.zeros(robots_num)
        target_line = np.array([target_x, target_y]).T
        line_similarity = evaluate_shape_similarity(
            run_result, target_shape=target_line
        )
        return line_similarity

    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        return [
            ("procrustes_distance", operator.lt, 0.1),
        ]
