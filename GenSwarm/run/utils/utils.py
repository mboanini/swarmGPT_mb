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

import numpy as np


def calculate_overlap_ratio(pos1, pos2, size1, size2, tolerance) -> float | None:
    """
    Calculate the overlap ratio between two entities if they are colliding.
    :param pos1: Position of the first entity
    :param pos2: Position of the second entity
    :param size1: Size of the first entity
    :param size2: Size of the second entity
    :param tolerance: Allowed tolerance for collision
    :return: overlap ratio if colliding, otherwise None
    """
    distance = np.linalg.norm(pos1 - pos2)
    combined_size = size1 + size2
    if distance + tolerance < combined_size:
        overlap = combined_size - distance
        return overlap / combined_size
    return None
