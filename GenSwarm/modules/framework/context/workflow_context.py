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

import argparse
import pickle

# from transformers import SEWDModel

from modules.file import File
from modules.framework.code import FunctionTree
from modules.framework.constraint import ConstraintPool

from .context import Context
from modules.prompt import robot_api


class WorkflowContext(Context):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)

        return cls._instance

    def _initialize(self,args=None):
        self.user_command = File(name="command.md")
        self.feedbacks = []
        self.run_code = File(name="run.py")
        self.args = args or argparse.Namespace()  # ✅ 使用传入的 args
        self._constraint_pool = ConstraintPool()
        # TODO:所有的命名统一化，比如这里的global skill tree和local skill tree (@Jiwenkang 10-4)
        if args:
            task_name = self.args.run_experiment_name[0]
            self.global_robot_api = robot_api.get_api_prompt(task_name, scope="global")
            self.local_robot_api = robot_api.get_api_prompt(task_name, scope="local")
            global_import_list = robot_api.get_api_prompt(
                task_name, scope="global", only_names=True
            )
            local_import_list = robot_api.get_api_prompt(task_name, scope="local", only_names=True)
            self.local_import_list = (
                local_import_list.split("\n\n")
                if isinstance(local_import_list, str)
                else local_import_list
            )
            self.local_import_list.append("get_assigned_task")
            self._global_skill_tree = FunctionTree(
                name="global_skill",
                init_import_list={
                    f"from global_apis import {','.join(global_import_list)}"
                },
            )
            self._local_skill_tree = FunctionTree(
                name="local_skill",
                init_import_list={
                    f"from apis import initialize_ros_node, {','.join(self.local_import_list)}"
                },
            )
        self.global_run_result = File(name="allocate_result.pkl")
        self.scoop = "global"
        self.vlm = False

    def save_to_file(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self._instance, file)
    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "rb") as file:
            instance = pickle.load(file)
            cls._instance = instance
            return instance

    def set_root_for_files(self, root_value):
        for file_attr in vars(self).values():
            if isinstance(file_attr, File):
                file_attr.root = root_value
            if isinstance(file_attr, FunctionTree):
                file_attr.file.root = root_value

    @property
    def command(self):
        return self._instance.user_command.message

    @command.setter
    def command(self, value):
        self._instance.user_command.message = value

    @property
    def global_skill_tree(self) -> FunctionTree:
        return self._instance._global_skill_tree

    @property
    def local_skill_tree(self) -> FunctionTree:
        return self._instance._local_skill_tree

    @property
    def constraint_pool(self) -> ConstraintPool:
        return self._instance._constraint_pool
