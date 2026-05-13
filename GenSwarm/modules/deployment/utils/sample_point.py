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
import pygame


def sample_point(
    zone_center=(0, 0),
    zone_shape="circle",
    zone_size=None,
    robot_size=None,
    robot_shape: str = "circle",
    min_distance: float = 0,
    entities=None,
    max_attempts_per_point=1000000,
):
    center = np.array(zone_center)

    if zone_shape == "circle":
        if zone_size is None or len(zone_size) != 1:
            raise ValueError(
                "For circle, size should be a list or tuple with one element: [radius]."
            )
        zone_radius = zone_size[0] - 2 * robot_size
    elif zone_shape == "rectangle":
        if zone_size is None or len(zone_size) != 2:
            raise ValueError(
                "For rectangle, size should be a list or tuple with two elements: [width, height]."
            )
        zone_width, zone_height = zone_size
        zone_width -= 2 * robot_size
        zone_height -= 2 * robot_size
    else:
        raise ValueError(f"Unsupported shape: {zone_shape}")

    collide_flag = True
    while max_attempts_per_point and collide_flag == True:
        if zone_shape == "circle":
            # Generate random points within the bounding rectangle and then filter to within the circle
            new_random_point = (
                np.random.uniform(-zone_radius, zone_radius, size=(2,)) + center
            )
            valid_mask = np.linalg.norm(new_random_point - center) <= zone_radius
            new_random_point = new_random_point[valid_mask]
        elif zone_shape == "rectangle":
            # Generate random points within the bounding rectangle
            new_random_point = (
                np.random.uniform(-0.5, 0.5, size=(2,))
                * np.array([zone_width, zone_height])
                + center
            )
        else:
            raise ValueError(f"Unsupported shape: {zone_shape}")

        collide_flag = check_collide(
            new_random_point, robot_size + min_distance, robot_shape, entities
        )

        max_attempts_per_point -= 1
        if max_attempts_per_point == 0:
            raise ValueError("Failed to find a valid point after maximum attempts.")

    return np.array(new_random_point)


def check_collide(position, size, shape, entities):
    for entity in entities:
        if shape == "circle":
            if entity.shape == "circle":
                if np.linalg.norm(position - entity.position) < (size + entity.size):
                    return True

            # elif entity.shape == 'rectangle':
            #     if check_circle_rectangle_collision(entity.position, position, entity.size, size * 2):
            #         return True

        # elif shape == 'rectangle':
        #     if entity.shape == 'circle':
        #         if check_circle_rectangle_collision(position, entity.position, size * 2, entity.size):
        #             return True
        #     else:
        #         rect1 = pygame.Rect(position[0] - size[0] / 2, position[1] - size[1] / 2, size[0], size[1])
        #         rect2 = pygame.Rect(entity.position[0] - entity.position[0] / 2,
        #                             entity.position[1] - entity.position[1] / 2,
        #                             entity.position[0], entity.size2[1])
        #         if rect1.colliderect(rect2):
        #             return True

    return False


def check_circle_rectangle_collision(
    rect_position, circle_position, rect_size, circle_radius
):
    if rect_size[0] / 2 + circle_radius < abs(rect_position[0] - circle_position[0]):
        return False
    elif rect_size[1] / 2 + circle_radius < abs(rect_position[1] - circle_position[1]):
        return False
    else:
        return True
