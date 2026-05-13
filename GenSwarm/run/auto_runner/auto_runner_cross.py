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
from modules.deployment.gymnasium_env import GymnasiumCrossingEnvironment
from run.auto_runner import AutoRunnerBase
from run.utils import check_collisions, evaluate_target_achievement


class AutoRunnerCross(AutoRunnerBase):
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
        test_mode="full_version",
        tolerance=0.05,
    ):
        env = GymnasiumCrossingEnvironment(env_config_path, radius=2.20)
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
            env=env,
        )

    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        return [
            ("target_achievement_ratio", operator.ge, 0.99),
        ]

    def analyze_result(self, run_result):
        collisions = check_collisions(run_result, tolerance=self.tolerance)
        target_achievement = evaluate_target_achievement(
            run_result, tolerance=self.tolerance
        )
        return collisions | target_achievement


if __name__ == "__main__":
    runner = AutoRunnerCross(
        "../../config/env/crossing_config.json",
        workspace_path="ablation/human_feedback",
        # workspace_path='metagpt',
        # workspace_path='cap/cross',
        experiment_duration=30,
        run_mode="rerun",
        # target_pkl='video_critic.pkl',
        target_pkl="None",
        # script_name='run_meta.py',
        script_name="run_cap.py",
        max_speed=0.75,
        tolerance=0.15,
    )
    # 人工复核，哪些任务需要重新跑，写在下面
    # experiment_list = ['2024-07-21_18-26-12', ]
    experiment_list = None
