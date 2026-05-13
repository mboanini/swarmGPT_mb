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


def extra_exp(base_path, out_type="dir"):
    if out_type == "dir":
        return [
            os.path.join(base_path, item)
            for item in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, item))
            and os.path.exists(os.path.join(base_path, item, "human_feedback.txt"))
        ]
    elif out_type == "name":
        return [
            item
            for item in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, item))
            and os.path.exists(os.path.join(base_path, item, "human_feedback.txt"))
        ]
