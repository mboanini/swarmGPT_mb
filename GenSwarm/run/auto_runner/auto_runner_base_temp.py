import asyncio
import json
import operator
import os
import queue
import subprocess
import threading
import time
import traceback
from abc import ABC, abstractmethod

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import rospy
from tqdm import tqdm
import concurrent.futures
from modules.deployment.gymnasium_env import GymnasiumEnvironmentBase
from modules.deployment.utils.manager import Manager
from modules.file import logger
from modules.framework.actions.run_code import runcode
from run.utils import setup_metagpt, setup_cap, check_robots_no_movement_in_first_third


class AutoRunnerBase(ABC):
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
        env: GymnasiumEnvironmentBase = None,
    ):
        self.env_config_path = env_config_path
        self.experiment_path = workspace_path
        self.experiment_duration = experiment_duration
        self.env = env
        self.env.reset()
        self.results = {}
        self.target_pkl = target_pkl
        self.manager = Manager(self.env, max_speed=max_speed)
        self.script_name = script_name
        self.stop_event = threading.Event()
        self.run_mode = run_mode
        self.tolerance = tolerance
        self.frames = []
        self.success_conditions = self.setup_success_conditions()

    def get_experiment_directories(self):
        directories = []
        for item in os.listdir(f"../workspace/{self.experiment_path}"):
            if item == "pic":
                continue
            item_path = os.path.join(f"../workspace/{self.experiment_path}", item)
            if os.path.isdir(item_path):
                if self.run_mode == "continue":
                    if not self.experiment_completed(item_path):
                        directories.append(item)
                elif self.run_mode == "rerun":
                    directories.append(item)
                elif self.run_mode == "analyze":
                    if self.experiment_completed(item_path):
                        directories.append(item)
        return directories

    def experiment_completed(self, path):
        result_file = os.path.join(path, "result.json")
        return os.path.exists(result_file)

    def save_experiment_result(
        self, path, result, analysis, run_code_result, retry_times
    ):
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        serializable_result = convert_to_serializable(result)
        serializable_analysis = convert_to_serializable(analysis)
        combined_result = {
            "run_code_result": run_code_result,
            "retry_times": retry_times,
            "analysis": serializable_analysis,
            "experiment_data": serializable_result,
        }
        result_file = os.path.join(path, "result.json")
        os.makedirs(path, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(combined_result, f, indent=4)

    def save_frame_to_disk(self, frame, frame_index):
        frame_path = os.path.join(self.frame_dir, f"frame_{frame_index:06d}.png")
        imageio.imwrite(frame_path, frame)

    def save_frames_as_animations(self, experiment_id):
        # Save as GIF
        gif_path = os.path.join(
            f"../workspace/{self.experiment_path}", experiment_id, "animation.gif"
        )
        imageio.mimsave(gif_path, self.frames, fps=10)
        print(f"Saved animation for experiment {experiment_id} as GIF at {gif_path}")

        # Save as MP4
        mp4_path = os.path.join(
            f"../workspace/{self.experiment_path}", experiment_id, "animation.mp4"
        )
        height, width, layers = self.frames[0].shape
        size = (width, height)
        out = cv2.VideoWriter(
            mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), self.env.FPS, size
        )

        for frame in self.frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"Saved animation for experiment {experiment_id} as MP4 at {mp4_path}")

        self.frames.clear()

    def run_single_experiment(self, experiment_id, result_queue):
        obs, infos = self.env.reset()

        def init_result(infos):
            result = {}
            for entity_id in infos:
                result[entity_id] = {
                    "size": 0,
                    "target": None,
                    "trajectory": [],
                    "type": "",
                    "states": [],
                    "dt": self.env.dt,
                }
                result[entity_id]["size"] = infos[entity_id]["size"]
                result[entity_id]["target"] = infos[entity_id]["target_position"]
                result[entity_id]["type"] = infos[entity_id]["type"]
                result[entity_id]["trajectory"].append(infos[entity_id]["position"])
            return result

        result = init_result(infos)
        self.manager.publish_observations(infos)
        rate = rospy.Rate(self.env.FPS)
        start_time = rospy.get_time()
        experiment_path = os.path.join(
            f"../workspace/{self.experiment_path}", experiment_id
        )

        self.frame_dir = os.path.join(experiment_path, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)

        frame_index = 0
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            current_time = rospy.get_time()
            if current_time - start_time > self.experiment_duration - 2:
                break
            action = self.manager.robotID_velocity
            self.manager.clear_velocity()
            obs, reward, termination, truncation, infos = self.env.step(action=action)
            for entity_id in infos:
                if infos[entity_id]["moveable"]:
                    result[entity_id]["trajectory"].append(infos[entity_id]["position"])
                if infos[entity_id]["state"] is not None:
                    result[entity_id]["states"].append(infos[entity_id]["state"])
            self.frames.append(self.env.render())
            self.manager.publish_observations(infos)
            rate.sleep()

        print(f"Experiment {experiment_id} completed successfully.")
        result_queue.put({"source": "run_single_experiment", "result": result})

    def run_code(self, experiment, script_name, result_queue):
        experiment_path = os.path.join(self.experiment_path, experiment)

        try:
            # 调用该函数并传入所需的参数（此处为同步函数，直接调用即可）
            result = runcode(
                experiment_path=experiment_path,
                timeout=self.experiment_duration - 3,
                target_pkl=self.target_pkl,
                feedback="VLM",
                script=script_name,
            )

            # 处理结果，假设 `result` 是执行后的返回值
            if result.get("success"):
                result_queue.put({"source": "run_code", "error": False, "reason": ""})
            else:
                result_queue.put(
                    {
                        "source": "run_code",
                        "error": True,
                        "reason": result.get("error_message", ""),
                    }
                )

        except Exception as e:
            traceback.print_exc()
            print(f"Error in run_code: {e}")
            result_queue.put({"source": "run_code", "error": True, "reason": str(e)})

    def run_multiple_experiments(self, experiment_list):
        if not experiment_list:
            experiment_list = sorted(self.get_experiment_directories())

        try:
            with tqdm(total=len(experiment_list), desc="Running Experiments") as pbar:
                for experiment in experiment_list:
                    retries = 0
                    max_retries = 1
                    success = False

                    while retries < max_retries and not success:
                        if self.script_name == "run_meta.py":
                            setup_metagpt(
                                os.path.join(
                                    f"../workspace/{self.experiment_path}", experiment
                                )
                            )
                        if self.script_name == "run_cap.py":
                            setup_cap(
                                os.path.join(
                                    f"../workspace/{self.experiment_path}", experiment
                                )
                            )

                        self.stop_event.clear()
                        result_queue = queue.Queue()

                        # 使用线程池来并发运行两个函数
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=2
                        ) as executor:
                            # 启动线程1：运行实验代码
                            t1 = executor.submit(
                                self.run_code,
                                experiment,
                                self.script_name,
                                result_queue,
                            )
                            # 启动线程2：监控机器人运动情况
                            t2 = executor.submit(
                                self.run_single_experiment, experiment, result_queue
                            )

                            # 等待两个线程完成
                            concurrent.futures.wait([t1, t2])

                        # 检查实验结果
                        single_experiment_result = None
                        run_code_result = None
                        while not result_queue.empty():
                            result = result_queue.get()
                            if result["source"] == "run_single_experiment":
                                single_experiment_result = result["result"]
                            elif result["source"] == "run_code":
                                run_code_result = result

                        analysis = self.analyze_result(single_experiment_result)
                        print(f"Analysis for experiment {experiment}: {analysis}")
                        experiment_success = self.calculate_success(analysis)
                        success_dict = {"success": experiment_success}
                        analysis.update(success_dict)

                        success = True  # 假设实验成功

                        self.results[experiment] = analysis
                        self.save_experiment_result(
                            os.path.join(
                                f"../workspace/{self.experiment_path}", experiment
                            ),
                            single_experiment_result,
                            analysis,
                            run_code_result,
                            retries,
                        )
                        if not success:
                            print(
                                f"Experiment {experiment} failed after {max_retries} attempts, moving to next experiment."
                            )
                        else:
                            print(f"Experiment {experiment} completed successfully.")
                        self.save_frames_as_animations(experiment)

                        pbar.update(1)

        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping all experiments.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_and_print_results(
        self,
        data,
        labels,
        ylabel,
        title,
        colors,
        save_filename,
        rotation=False,
        success_conditions=None,
    ):
        if rotation:
            rotation = -90
        else:
            rotation = 0

        plt.figure(figsize=(10, 6))
        plt.bar(labels, data, color=colors)
        plt.xticks(rotation=rotation)
        plt.ylabel(ylabel)
        plt.title(title)

        # 标记成功条件线
        if success_conditions:
            for condition in success_conditions:
                metric, operator, threshold = condition
                if metric == ylabel:  # 仅在当前绘制的 metric 与成功条件相关时绘制
                    plt.axhline(
                        y=threshold,
                        color="r",
                        linestyle="--",
                        label=f"Success threshold ({threshold})",
                    )

        for i, v in enumerate(data):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center", rotation=rotation)

        if not os.path.exists(f"../workspace/{self.experiment_path}/pic"):
            os.makedirs(f"../workspace/{self.experiment_path}/pic")
        plt_path = os.path.join(
            f"../workspace/{self.experiment_path}/pic", save_filename
        )
        plt.tight_layout()
        plt.legend()
        plt.savefig(plt_path)
        plt.show()
        print(f"{title}: {data}")

    def analyze_code(self, functions_path):
        from modules.utils import CodeAnalyzer

        with open(functions_path, "r") as f:
            analyzer = CodeAnalyzer(f.read())
            analysis_result = analyzer.analyze()
        # TODO:add more metrics
        return analysis_result["code_lines"]

    def run(self, exp_list=None):
        if self.run_mode in ["rerun", "continue"]:
            self.run_multiple_experiments(exp_list)

        if self.run_mode == "analyze":
            self.analyze_all_results(exp_list)

    @abstractmethod
    def analyze_result(self, run_result) -> dict[str, float]:
        """
        Analyze the result of a single experiment and return a dictionary of metrics.
        """
        raise NotImplementedError(
            "analyze_result method must be implemented in the subclass"
        )

    @abstractmethod
    def setup_success_conditions(self) -> list[tuple[str, operator, float]]:
        """
        This method should be implemented in the subclass to set up the success conditions for the experiments.
        The success conditions should be a list of tuples, where each tuple contains the metric name, the comparison
        operator, and the threshold value. For example, to set up success conditions for a metric "mean_dtw_distance"
        with a threshold of 500, the method should return a list like this:
        [("mean_dtw_distance", operator.lt, 500)]
        """
        raise NotImplementedError(
            "setup_success_conditions method must be implemented in the subclass"
        )

    def calculate_success(self, analysis):
        success = True
        for condition in self.success_conditions:
            metric, operator, threshold = condition
            if metric not in analysis:
                print(f"Metric {metric} not found in analysis.")
                continue
            if not operator(analysis[metric], threshold):
                success = False
                break
        return success

    def analyze_all_results(self, experiment_dirs=None):
        if not experiment_dirs:
            experiment_dirs = sorted(self.get_experiment_directories())

        exp_data = {experiment: {} for experiment in experiment_dirs}
        all_metric_names = []

        for experiment in experiment_dirs:
            result_path = os.path.join(
                f"../workspace/{self.experiment_path}", experiment, "result.json"
            )
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    result_data = json.load(f)
                    analysis = result_data.get("analysis", {})
                    bug = result_data.get("run_code_result", {})["error"]
                    success = self.calculate_success(analysis)
                    data = {
                        **analysis,
                        "success": success,
                        "BUG": bug,
                        "retry_times": result_data.get("retry_times", 0),
                    }
                    exp_data[experiment] = data

                    # Collect all possible metric keys from analysis
                    all_metric_names = list(data.keys())

        # Calculate and plot average metrics
        mean_metric_value = {
            metric: np.mean([exp_data[exp].get(metric, 0) for exp in exp_data.keys()])
            for metric in all_metric_names
        }
        for metric in all_metric_names:
            self.plot_and_print_results(
                data=[exp_data[exp].get(metric, 0) for exp in exp_data.keys()],
                labels=exp_data.keys(),
                ylabel=metric,
                title=f"Experiment Outcomes in {metric}",
                colors=[self.get_color(i, len(exp_data)) for i in range(len(exp_data))],
                save_filename=f"{metric}_metric.png",
                rotation=True,
                success_conditions=self.success_conditions,  # 传递成功条件
            )

        # Plot summary of metrics
        self.plot_summary(mean_metric_value)

    def plot_summary(self, exp_data):
        """
        This method plots all the metrics on a single graph, rotating x-axis labels by 90 degrees,
        and uses plot_and_print_results to handle the plotting.
        """
        # Prepare data and labels from exp_data
        labels = list(exp_data.keys())  # Names of the metrics
        data = list(exp_data.values())  # Average values of the metrics

        # Since there's only one set of data (the averages), we don't need multiple colors
        colors = ["blue"]  # You can choose a single color for all bars

        # Call plot_and_print_results to plot all averages in one graph
        self.plot_and_print_results(
            data=data,
            labels=labels,
            ylabel="Average Value",
            title="Summary of All Metric Averages",
            colors=colors,  # Single color for all metrics
            save_filename="summary_metrics.png",
            rotation=False,  # Rotate the x-axis labels 90 degrees for readability
        )

    @staticmethod
    def get_color(index, total):
        cmap = plt.get_cmap("viridis")
        return cmap(index / total)
