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

from .prey import Prey
import numpy as np


class Sheep(Prey):
    def __init__(
        self,
        prey_id,
        initial_position,
        size,
        max_speed=1,
        danger_zone=1,
        damping=0.9,
        random_factor=0.1,
        alpha=0.1,
        density=0.1,
        mass=1.0,
    ):
        super().__init__(
            prey_id,
            initial_position,
            size,
            mass=mass,
            density=density,
            max_speed=max_speed,
            danger_zone=danger_zone,
            damping=damping,
            random_factor=random_factor,
            alpha=alpha,
        )

    def calculate_velocity(self, flock, robots, environment_bounds):
        random_movement = np.random.randn(2) * self.random_factor  # 随机游走
        repulsion = self.separate(flock + robots)  # 分离
        avoid_edge = self.avoid_edges(environment_bounds=environment_bounds)  # 边缘避免行为
        avoid_dogs = self.avoid_dogs(flock=flock, robots=robots)  # 往羊群钻
        new_velocity = (
            1 * self.velocity
            + 2 * repulsion  # 惯性速度
            +
            # 1 * avoid_edge +
            2 * avoid_dogs
            + 1 * random_movement
        )

        self.filtered_velocity = (
            self.alpha * new_velocity + (1 - self.alpha) * self.filtered_velocity
        )

        speed = np.linalg.norm(self.filtered_velocity)
        if speed > self.max_speed:
            self.filtered_velocity = self.filtered_velocity / speed * self.max_speed

        self.velocity = self.filtered_velocity
        return self.velocity

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

    def avoid_dogs(self, robots, flock):
        new_velocity = 0
        if any(
            [
                np.linalg.norm(self.position - dog.position) < self.danger_zone
                for dog in robots
            ]
        ):
            flock_center = np.mean([sheep.position for sheep in flock], axis=0)
            direction_to_flock = flock_center - self.position
            distance_to_flock = np.linalg.norm(direction_to_flock)
            distance_to_dog = min(
                [np.linalg.norm(self.position - dog.position) for dog in robots]
            )

            direction_to_flock_normalized = (
                direction_to_flock / distance_to_flock
                if distance_to_flock != 0
                else np.zeros(2)
            )
            speed_factor = distance_to_dog / (distance_to_flock + 1e-5)
            new_velocity = direction_to_flock_normalized * speed_factor * self.max_speed
        return new_velocity
