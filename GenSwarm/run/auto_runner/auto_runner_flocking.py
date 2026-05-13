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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import operator
from modules.deployment.gymnasium_env import GymnasiumFlockingEnvironment
from run.auto_runner import AutoRunnerBase
from run.utils import (
    evaluate_trajectory_similarity,
    evaluate_trajectory_pattern,
    check_collisions,
    evaluate_min_distances_to_others,
    evaluate_average_position, check_robots_no_movement_in_first_third,
)


class AutoRunnerFlocking(AutoRunnerBase):
    def __init__(
            self,
            env_config_path,
            workspace_path,
            experiment_duration,
            run_mode="rerun",
            target_pkl="WriteRun.pkl",
            script_name="run.py",
            test_mode=None,
            exp_batch=1,
            max_speed=1.0,
            tolerance=0.05,
    ):
        env = GymnasiumFlockingEnvironment(env_config_path)
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

    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        return [
            ("spatial_variance", operator.lt, 1.0),
            ("mean_dtw_distance", operator.lt, 800),
            # ("no_move_num", operator.lt, 1),
            # ("max_min_distance", operator.lt, 1),
            # ("average_distance", operator.lt, 0.5),
        ]

    def analyze_result(self, run_result) -> dict[str, float]:
        terminal_distance = evaluate_trajectory_pattern(run_result)
        no_move_num = check_robots_no_movement_in_first_third(run_result)
        max_min_distance = evaluate_min_distances_to_others(run_result)
        average_distance = evaluate_average_position(run_result)
        similarity = evaluate_trajectory_similarity(run_result)
        collision = check_collisions(run_result)
        merged_dict = (
                max_min_distance
                | terminal_distance
                | similarity
                | collision
                | average_distance | no_move_num
        )
        return merged_dict
