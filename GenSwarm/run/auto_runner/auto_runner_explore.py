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
import os
import json
from abc import ABC

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import operator
from modules.deployment.gymnasium_env import GymnasiumExplorationEnvironment
from run.auto_runner import AutoRunnerBase
from run.utils import evaluate_landmark_visits, check_collisions


class AutoRunnerExplore(AutoRunnerBase):
    def __init__(
        self,
        env_config_path,
        workspace_path,
        experiment_duration,
        run_mode="rerun",
        target_pkl="WriteRun.pkl",
        script_name="run.py",
        max_speed=1.0,
        exp_batch=1,
        tolerance=0.05,
        test_mode=None,
    ):
        env = GymnasiumExplorationEnvironment(env_config_path)
        super().__init__(
            env_config_path=env_config_path,
            workspace_path=workspace_path,
            experiment_duration=experiment_duration,
            run_mode=run_mode,
            target_pkl=target_pkl,
            script_name=script_name,
            max_speed=max_speed,
            tolerance=tolerance,
            test_mode=test_mode,
            exp_batch=exp_batch,
            env=env,
        )

    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        return [
            ("landmark_visit_ratio", operator.ge, 0.99),
        ]

    def analyze_result(self, run_result) -> dict[str, float]:
        landmark_visit = evaluate_landmark_visits(run_result)
        collisions = check_collisions(run_result, tolerance=self.tolerance)
        return landmark_visit | collisions
