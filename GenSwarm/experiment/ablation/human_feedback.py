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

from modules.framework.action import ActionNode
from modules.framework.context import WorkflowContext
from .base_alation import BaseAlation
from modules.framework.actions import Criticize


class HumanFeedbackAlation(BaseAlation):
    def __init__(self, workspace_name: str, exp_list: list):
        criticizer = Criticize()
        super().__init__(criticizer, workspace_name, exp_list)

    def setup(self, directory: str):
        self.tester = Criticize()
        self.tester.context.load_from_file(f"{directory}/WriteRun.pkl")
        feedback_file_path = f"{directory}/human_feedback.txt"
        with open(feedback_file_path, "r") as file:
            feedback = file.read()
        self.tester.setup(feedback)

    async def run_single(self, directory: str):
        await self.tester.run()
        self.tester.context.save_to_file(f"{directory}/human_feedback.pkl")
