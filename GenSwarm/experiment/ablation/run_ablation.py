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

from experiment.ablation import VideoCriticAlation, HumanFeedbackAlation
from experiment.ablation.utils import extra_exp


async def main(exp_type: str, workspace_name: str, exp_list: list):
    if exp_type == "video_critic":
        tester = VideoCriticAlation(workspace_name, exp_list)
    elif exp_type == "human_feedback":
        tester = HumanFeedbackAlation(workspace_name, exp_list)

    await tester.run_exp()


if __name__ == "__main__":
    import asyncio

    exp_type = "video_critic"
    # exp_type = "human_feedback"
    exp_list = sorted(extra_exp("../../workspace/parallel/cross"))
    # exp_list = sorted("../../workspace/layer/cross")

    workspace_name = f"../../workspace/ablation/layer/{exp_type}"

    asyncio.run(
        main(exp_type=exp_type, workspace_name=workspace_name, exp_list=exp_list)
    )
