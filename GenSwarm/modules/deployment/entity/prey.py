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

from .base_entity import Entity
from ..utils.traectory_generator import generate_arc_trajectory


class Prey(Entity):
    def __init__(self, prey_id, initial_position, size, num=2000):
        super().__init__(
            prey_id,
            initial_position,
            size,
            color="blue",
            collision=True,
            movable=True,
            max_speed=1.0,
            mass=1,
            density=1,
        )
        self.move_mode = "track"
        self.target_trajectory = generate_arc_trajectory(num, 0, 12.28, 1.5, (0, 0))
        self.position = self.target_trajectory[0]
        self.max_speed = 1.0
        self.random_factor = 0.1
        self.velocity = np.zeros(2)
        self.filtered_velocity = np.zeros(2)
        self.alpha = 0

    def move_to_target(self, time_step):
        # print(f"prey move to target: {self.target_trajectory[time_step]},step:{time_step}")
        # self.velocity = self.target_trajectory[time_step] - self.position
        # self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        if time_step < len(self.target_trajectory):
            self.position = self.target_trajectory[time_step]

    def move(self, time_step):
        if self.move_mode == "track":
            self.move_to_target(time_step)

    def calculate_velocity(self, flock, robots, environment_bounds):
        random_movement = np.random.randn(2) * self.random_factor  # 随机游走
        repulsion = self.separate(flock + robots)  # 分离
        avoid_edge = self.avoid_edges(environment_bounds=environment_bounds)  # 边缘避免行为
        new_velocity = (
            1 * self.velocity
            + 2 * repulsion  # 惯性速度
            +
            # 1 * avoid_edge +
            1 * random_movement
        )

        self.filtered_velocity = (
            self.alpha * new_velocity + (1 - self.alpha) * self.filtered_velocity
        )

        speed = np.linalg.norm(self.filtered_velocity)
        if speed > self.max_speed:
            self.filtered_velocity = self.filtered_velocity / speed * self.max_speed

        self.velocity = self.filtered_velocity
        return self.velocity

    def separate(self, others):
        if len(others) == 0:
            return np.zeros(2)
        separation_force = np.zeros(2)
        for other in others:
            distance = np.linalg.norm(self.position - other.position)
            if distance < 1:  # 0.5m 范围内的排斥力
                separation_force -= (other.position - self.position) / (distance + 1e-5)
        if any(separation_force) != 0:
            separation_force = separation_force / np.linalg.norm(separation_force)
        return separation_force

    def avoid_edges(self, environment_bounds):
        avoidance_force = np.zeros(2)
        x, y = self.position

        if x < environment_bounds[0] + 0.5:
            avoidance_force[1] = 0.5 * (
                0.5 / (x - environment_bounds[0] + 1e-5)
            )  # 加速因子
            avoidance_force[0] = -avoidance_force[0]
        elif x > environment_bounds[1] - 0.5:
            avoidance_force[1] = -0.5 * (
                0.5 / (environment_bounds[1] - x + 1e-5)
            )  # 加速因子
            avoidance_force[0] = -avoidance_force[0]

        if y < environment_bounds[2] + 0.5:
            avoidance_force[0] = 0.5 * (
                0.5 / (y - environment_bounds[2] + 1e-5)
            )  # 加速因子
            avoidance_force[1] = -avoidance_force[1]

        elif y > environment_bounds[3] - 0.5:
            avoidance_force[0] = -0.5 * (
                0.5 / (environment_bounds[3] - y + 1e-5)
            )  # 加速因子
            avoidance_force[1] = -avoidance_force[1]

        if np.any(avoidance_force) != 0:
            avoidance_force = avoidance_force / np.linalg.norm(avoidance_force)
        return avoidance_force
