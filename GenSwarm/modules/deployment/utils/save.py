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

import cv2
import imageio


def save_frames_as_animations(id, experiment_path, frames):
    # Save as GIF
    gif_path = os.path.join(f"../workspace/{experiment_path}", f"animation_{id}.gif")
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Saved animation for experiment  as GIF at {gif_path}")

    # Save as MP4
    mp4_path = os.path.join(f"../workspace/{experiment_path}", f"animation_{id}.mp4")
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, size)

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Saved animation for experiment  as MP4 at {mp4_path}")

    frames.clear()
