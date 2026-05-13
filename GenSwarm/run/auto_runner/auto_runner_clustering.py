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

import math
import operator
import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from modules.deployment.gymnasium_env import (
    GymnasiumFlockingEnvironment,
    GymnasiumClusteringEnvironment,
)
from run.auto_runner import AutoRunnerBase
from run.utils import (
    evaluate_trajectory_similarity,
    evaluate_trajectory_pattern,
    evaluate_robot_quadrant_positions,
)


class AutoRunnerClustering(AutoRunnerBase):
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
        test_mode=None,
        tolerance=0.05,
    ):
        env = GymnasiumClusteringEnvironment(env_config_path)
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
            test_mode=test_mode,
            env=env,
        )

    def analyze_result(self, run_result) -> dict[str, float]:
        target_regions = {
            1: (0.5, 2, 0.5, 2),
            2: (-2, -0.5, 0.5, 2),
            3: (-2, -0.5, -2, -0.5),
            4: (0.5, 2, -2, -0.5)
            # 3: np.array([-1.25, -1.25]),
            # 2: np.array([-1.25, 1.25]),
            # 1: np.array([1.25, 1.25]),
            # 4: np.array([1.25, -1.25]),
        }
        similarity = evaluate_robot_quadrant_positions(
            run_result, target_regions=target_regions
        )
        return similarity

    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        return [
            ("achievement_ratio", operator.ge, 0.99),
        ]
