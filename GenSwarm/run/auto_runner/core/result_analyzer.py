import os
import json
import numpy as np
import matplotlib.pyplot as plt


# from streamlit import success


class ExperimentAnalyzer:
    def __init__(self, experiment_path, success_conditions, tolerance=0.05):
        self.experiment_path = experiment_path
        self.success_conditions = success_conditions

    def calculate_success(self, analysis):
        success = True
        for condition in self.success_conditions:
            metric, operator, threshold = condition
            if metric not in analysis:
                print(f"Metric {metric} not found in analysis.")
                return False
            if not operator(analysis[metric], threshold):
                success = False
                break
        return success

    def analyze_all_results(self, experiment_dirs=None, target_file=['wo_vlm.json'],
                            only_success: bool = False):
        exp_data = {experiment: {} for experiment in experiment_dirs}
        all_metric_names = []

        for experiment in experiment_dirs:
            combined_success = False  # Initialize combined success as False for each experiment
            first_file_data = None  # Variable to store the first file's data for non-success metrics
            analysis = {}
            for file in target_file:
                result_path = os.path.join(self.experiment_path, experiment, file)
                if os.path.exists(result_path):
                    with open(result_path, 'r') as f:
                        result_data = json.load(f)
                        if file == 'vlm.json':
                            success = result_data.get('success', False)
                            combined_success = combined_success or success
                        else:
                            analysis = result_data.get('analysis', {})
                            success = analysis.get('success', False)
                            combined_success = combined_success or success  # Logical OR with the existing combined success
                else:
                    print(f"File {result_path} not found.")
                    # if file == 'improve.json':
                    #     combined_success = True
            data = {**analysis, 'success': combined_success}

            exp_data[experiment] = data
            all_metric_names = list(data.keys())

        mean_metric_value = {}
        for metric in all_metric_names:
            metric_values = []
            for exp in exp_data.keys():
                value = exp_data[exp].get(metric, 0)
                if isinstance(value, (int, float)):
                    metric_values.append(value)
            if metric_values:
                mean_metric_value[metric] = np.mean(metric_values)
            else:
                mean_metric_value[metric] = 0

        # Plotting results
        for metric in all_metric_names:
            if only_success and metric != 'success':
                continue
            data = [
                exp_data[exp].get(metric, 0) if isinstance(exp_data[exp].get(metric, 0), (int, float)) else 0
                for exp in exp_data.keys()
            ]
            filename = f'{metric}_metric.png'

            if metric == 'success':
                filename = 'success_' + '_'.join(target_file) + '.png'
            self.plot_and_print_results(
                data=data,
                labels=exp_data.keys(),
                ylabel=metric,
                title=f"Experiment Outcomes in {metric}",
                colors=[self.get_color(i, len(exp_data)) for i in range(len(exp_data))],
                save_filename=filename,
                rotation=True,
                figsize=(32, 20),
                success_conditions=self.success_conditions
            )

        self.plot_summary(mean_metric_value, file_name='_'.join(target_file))

    def plot_summary(self, exp_data, file_name):
        labels = list(exp_data.keys())
        data = list(exp_data.values())
        colors = ['blue']

        self.plot_and_print_results(
            data=data,
            labels=labels,
            ylabel='Average Value',
            title='Summary of All Metric Averages',
            colors=colors,
            save_filename=f'summary_metrics_{file_name}.png',
            rotation=False,
            figsize=(32, 20)  # 传入图像大小参数
        )

    @staticmethod
    def get_color(index, total):
        cmap = plt.get_cmap('viridis')
        return cmap(index / total)

    def analyze_code(self, functions_path):
        from modules.utils import CodeAnalyzer
        with open(functions_path, 'r') as f:
            analyzer = CodeAnalyzer(f.read())
            analysis_result = analyzer.analyze()
        return analysis_result["code_lines"]

    def plot_and_print_results(self, data, labels, ylabel, title, colors, save_filename, rotation=False,
                               figsize=(10, 6), success_conditions=None):  # 添加figsize参数
        if rotation:
            rotation = -90
        else:
            rotation = 0

        plt.figure(figsize=figsize)  # 使用传入的figsize设置图像大小
        plt.bar(labels, data, color=colors)
        plt.xticks(rotation=rotation)
        plt.ylabel(ylabel)
        plt.title(title)

        if success_conditions:
            for condition in success_conditions:
                metric, operator, threshold = condition
                if metric == ylabel:
                    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Success threshold ({threshold})")

        for i, v in enumerate(data):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', rotation=rotation)

        if not os.path.exists(f"{self.experiment_path}/pic"):
            os.makedirs(f"{self.experiment_path}/pic")
        plt_path = os.path.join(f"{self.experiment_path}/pic", save_filename)
        txt_path = os.path.join(f"{self.experiment_path}/pic", save_filename.replace('.png', '.txt'))
        # 保存结果到 txt 文件
        with open(txt_path, 'w') as file:
            file.write("Summary of All Metric Averages\n")
            for label, value in zip(labels, data):
                file.write(f"{label}: {value}\n")

        plt.tight_layout()
        plt.legend()
        plt.savefig(plt_path)
        plt.show()
        print(f"{title}: {data}")
