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

from modules.deployment.gymnasium_env import (
    GymnasiumTransportationEnvironment,
    GymnasiumHerdingEnvironment,
)
from modules.deployment.gymnasium_env.gymnasium_formation_env import (
    GymnasiumFormationEnvironment,
)
from run.auto_runner import AutoRunnerBase


class AutoRunnerHerding(AutoRunnerBase):
    def __init__(
        self,
        env_config_path,
        workspace_path,
        experiment_duration,
        run_mode="rerun",
        target_pkl="WriteRun.pkl",
        script_name="run.py",
        max_speed=1.0,
        tolerance=0.05,
    ):
        env = GymnasiumHerdingEnvironment(env_config_path)
        super().__init__(
            env_config_path=env_config_path,
            workspace_path=workspace_path,
            experiment_duration=experiment_duration,
            run_mode=run_mode,
            target_pkl=target_pkl,
            script_name=script_name,
            max_speed=max_speed,
            tolerance=tolerance,
            env=env,
        )

    def analyze_result(self, run_result) -> dict[str, float]:
        pass

    def analyze_all_results(self, experiment_dirs=None):
        pass
