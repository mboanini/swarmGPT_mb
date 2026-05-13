import json
import os
import pickle
import traceback

from modules.utils.root import root_manager


class CodeRunner:
    def __init__(
            self,
            time_out,
            target_pkl,
            script_name,
            feedback,
            experiment_path: str,
            env_manager=None,
            test_mode=None,
    ):
        self.time_out = time_out
        self.target_pkl = target_pkl
        self.script_name = script_name
        self.feedback = feedback
        self.experiment_path = experiment_path
        self.env_manager = env_manager
        self.test_mode = test_mode

    def run_code(self, experiment_id):
        experiment_path = os.path.join(self.experiment_path, experiment_id)
        root_manager.update_root(experiment_path)
        try:
            from modules.framework.actions.run_code import runcode

            result = runcode(
                experiment_path=experiment_path,
                timeout=self.time_out,
                target_pkl=self.target_pkl,
                feedback=self.feedback,
                script=self.script_name,
                env_manager=self.env_manager,
                test_mode=self.test_mode,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Error in run_code: {e}")
            # result_queue.put({'source': 'run_code', 'error': True, 'reason': str(e)})

    def load_result(self, experiment_id, result_type):
        experiment_path = os.path.join(self.experiment_path, experiment_id)
        file_path = os.path.join(experiment_path, f'{result_type}.pkl')
        if os.path.exists(file_path):

            with open(file_path, "rb") as file:
                result = pickle.load(file)
        else:
            result = None

        return result

    def load_run_result(self, experiment_id, result_type):
        experiment_path = os.path.join(self.experiment_path, experiment_id)
        combined_results = {}

        for run_mode in ["global", "local", "init"]:
            file_path = os.path.join(
                experiment_path, f"{result_type}_{run_mode}_run.json"
            )

            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    result = json.load(f)
            else:
                result = None
            # delete the file
            if result is not None:
                os.remove(file_path)
            combined_results[run_mode] = result

        return combined_results
