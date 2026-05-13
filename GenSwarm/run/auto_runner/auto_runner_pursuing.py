import operator

from modules.deployment.gymnasium_env import GymnasiumPursuingEnvironment
from run.auto_runner import AutoRunnerBase
from run.utils import (
    evaluate_robot_prey_distance,
    evaluate_average_position,
    evaluate_trajectory_similarity,
    check_collisions,
    evaluate_min_distances_to_others,
    evaluate_average_distance_to_prey,
)


class AutoRunnerPursuing(AutoRunnerBase):
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
        env = GymnasiumPursuingEnvironment(env_config_path)
        super().__init__(
            env_config_path=env_config_path,
            workspace_path=workspace_path,
            experiment_duration=experiment_duration,
            run_mode=run_mode,
            target_pkl=target_pkl,
            script_name=script_name,
            max_speed=max_speed,
            exp_batch=exp_batch,
            tolerance=tolerance,
            test_mode=test_mode,
            env=env,
        )

    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        return [
            ("max_min_distance", operator.lt, 1),
            ("average_distance", operator.lt, 1),
        ]

    def analyze_result(self, run_result) -> dict[str, float]:
        max_min_distance = evaluate_min_distances_to_others(run_result)
        average_distance = evaluate_average_distance_to_prey(run_result)
        similarity = evaluate_trajectory_similarity(run_result)
        collision = check_collisions(run_result)
        merged_dict = max_min_distance | similarity | collision | average_distance
        return merged_dict
