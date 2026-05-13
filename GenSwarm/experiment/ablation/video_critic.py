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

from modules.framework.handler import FeedbackHandler
from .base_alation import BaseAlation
from modules.framework.actions import VideoCriticize, Criticize
from modules.utils import process_video, create_video_from_frames


class VideoCriticAlation(BaseAlation):
    def __init__(self, workspace_name: str, exp_list: list):
        criticizer = VideoCriticize()
        super().__init__(criticizer, workspace_name, exp_list)

    def setup(self, directory: str):
        self.tester = VideoCriticize()
        hf_handler = FeedbackHandler()
        criticize = Criticize("feedback")
        hf_handler.next_action = criticize
        self.tester.error_handler = hf_handler
        self.tester.context.load_from_file(f"{directory}/WriteRun.pkl")
        frames = process_video(
            f"{directory}/animation.mp4", end_time=30, seconds_per_frame=2
        )
        create_video_from_frames(
            base64Frames=frames, output_path=f"{directory}/extra.mp4", fps=10
        )
        self.tester.setup(frames)

    async def run_single(self, directory: str):
        await self.tester.run()
        self.tester.context.save_to_file(f"{directory}/video_critic.pkl")
