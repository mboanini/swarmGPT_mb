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
import shutil
from abc import ABC, abstractmethod

from tqdm import tqdm

from modules.framework.action import ActionNode
from modules.framework.context import WorkflowContext


class BaseAlation(ABC):
    def __init__(self, tester: ActionNode, workspace_name: str, exp_list: list):
        self._exp_list = exp_list
        self._tester = tester
        self._workspace = workspace_name

    @property
    def tester(self):
        return self._tester

    @tester.setter
    def tester(self, value):
        self._tester = value

    @property
    def workspace(self):
        return self._workspace

    @property
    def exp_list(self):
        return self._exp_list

    def init_workspace(self):
        new_exp_list = []
        for dir in self._exp_list:
            source_dir = dir
            destination_dir = os.path.join(self._workspace, os.path.basename(dir))
            try:
                if os.path.exists(destination_dir):
                    shutil.rmtree(
                        destination_dir
                    )  # Remove the destination directory if it exists
                shutil.copytree(source_dir, destination_dir)  # Copy the directory
                new_exp_list.append(destination_dir)
                print(f"Copied {source_dir} to {destination_dir}")
            except Exception as e:
                print(f"Failed to copy {source_dir} to {destination_dir}: {e}")
        self._exp_list = new_exp_list

    async def run_exp(self):
        self.init_workspace()
        with tqdm(
            total=len(self._exp_list), desc=f"Running {self.__class__.__name__}"
        ) as pbar:
            for dir in self._exp_list:
                self.setup(dir)
                await self.run_single(dir)
                pbar.update(1)

    @abstractmethod
    def setup(self, directory: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement setup method"
        )

    @abstractmethod
    async def run_single(self, directory: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement setup method"
        )
