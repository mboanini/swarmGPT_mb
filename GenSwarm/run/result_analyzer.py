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

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from modules.utils import CodeAnalyzer


class LogAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.data = {"parallel": [], "sequential": [], "layer": []}
        self.patterns = {
            "Generate functions cost time": re.compile(
                r"Generate functions cost time: (\d+\.\d+)"
            ),
            "average complexity": re.compile(
                r"average complexity: (\d+\.\d+), mi score: (\d+\.\d+)"
            ),
            "run code result": re.compile(r"run code:(success|fail)"),
        }
        self.total_files = 0
        self.missing_run_code_files = 0
        self.missing_files_list = []

    def analyze(self):
        # Traverse the directory structure
        for category in self.data.keys():
            category_path = os.path.join(self.base_path, category)
            if os.path.exists(category_path):
                for task in os.listdir(category_path):
                    task_path = os.path.join(category_path, task)
                    for experiment in os.listdir(task_path):
                        experiment_path = os.path.join(task_path, experiment)
                        if os.path.isdir(experiment_path):
                            log_file_path = os.path.join(experiment_path, "log.md")
                            function_path = os.path.join(
                                experiment_path, "functions.py"
                            )
                            if os.path.exists(log_file_path):
                                self._process_log_file(log_file_path, category)
                                self.total_files += 1
                            if os.path.exists(function_path):
                                self._process_functions_file(function_path, category)

    def _process_functions_file(self, function_file_path, category):
        with open(function_file_path, "r", encoding="utf-8") as file:
            function_contents = file.read()
        code_analyzer = CodeAnalyzer(code=function_contents)

    def _process_log_file(self, log_file_path, category):
        with open(log_file_path, "r", encoding="utf-8") as file:
            log_contents = file.read()

        extracted_data = {key: [] for key in self.patterns.keys()}
        extracted_data["mi score"] = []
        extracted_data["success"] = -1

        run_code_found = False

        for key, pattern in self.patterns.items():
            matches = pattern.findall(log_contents)
            if key == "average complexity":
                for match in matches:
                    extracted_data["average complexity"].append(float(match[0]))
                    extracted_data["mi score"].append(float(match[1]))
            elif key == "run code result":
                if matches:
                    extracted_data["success"] = 1 if matches[-1] == "success" else 0
                    run_code_found = True
            else:
                extracted_data[key].extend([float(match) for match in matches])

        if not run_code_found:
            self.missing_run_code_files += 1
            self.missing_files_list.append(log_file_path)

        self.data[category].append(extracted_data)

    def calculate_statistics(self):
        stats = {}
        for category, data_list in self.data.items():
            if data_list:
                category_data = {
                    key: np.array([item[key] for item in data_list if item[key]])
                    for key in self.patterns.keys()
                }
                category_data["mi score"] = np.array(
                    [item["mi score"] for item in data_list if item["mi score"]]
                )
                success_rate = (
                    np.mean(
                        [item["success"] for item in data_list if item["success"] != -1]
                    )
                    * 100
                )

                count = {key: len(values) for key, values in category_data.items()}
                mean = {
                    key: np.mean(values) if len(values) > 0 else np.nan
                    for key, values in category_data.items()
                }
                std = {
                    key: np.std(values) if len(values) > 0 else np.nan
                    for key, values in category_data.items()
                }

                stats[category] = {
                    "count": count,
                    "mean": mean,
                    "std": std,
                    "success_rate": success_rate,
                }
        return stats

    def plot_statistics(self, stats):
        if not stats:
            print("No data found to plot.")
            return

        metrics = ["Generate functions cost time", "average complexity", "mi score"]
        categories = list(stats.keys())

        mean_fig, mean_ax = plt.subplots(figsize=(10, 6))
        std_fig, std_ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.2

        for i, category in enumerate(categories):
            mean_values = [stats[category]["mean"][metric] for metric in metrics]
            std_values = [stats[category]["std"][metric] for metric in metrics]

            mean_ax.bar(x + i * width, mean_values, width, label=category)
            std_ax.bar(x + i * width, std_values, width, label=category)

        mean_ax.set_title("Mean Values")
        mean_ax.set_ylabel("Mean")
        mean_ax.set_xticks(x + width / 2)
        mean_ax.set_xticklabels(metrics)
        mean_ax.legend()

        std_ax.set_title("Standard Deviation Values")
        std_ax.set_ylabel("Standard Deviation")
        std_ax.set_xticks(x + width / 2)
        std_ax.set_xticklabels(metrics)
        std_ax.legend()

        plt.tight_layout()
        plt.show()

        success_rate_fig, success_rate_ax = plt.subplots(figsize=(10, 6))
        success_rates = [stats[category]["success_rate"] for category in categories]
        print(success_rates)
        success_rate_ax.bar(categories, success_rates, width)
        success_rate_ax.set_title("Success Rates")
        success_rate_ax.set_ylabel("Success Rate (%)")

        plt.tight_layout()
        plt.show()

    def print_summary(self):
        print(f"Total files: {self.total_files}")
        print(f"Files missing 'run code' results: {self.missing_run_code_files}")
        print(
            f"Percentage of files missing 'run code' results: {self.missing_run_code_files / self.total_files * 100:.2f}%"
        )
        if self.missing_files_list:
            print("Files missing 'run code' results:")
            for file in self.missing_files_list:
                print(file)


# Example usage
base_path = "../workspace"
analyzer = LogAnalyzer(base_path)
analyzer.analyze()
stats = analyzer.calculate_statistics()

if stats:
    analyzer.plot_statistics(stats)
else:
    print("No statistical data available.")
analyzer.print_summary()
