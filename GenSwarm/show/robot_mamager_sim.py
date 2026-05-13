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

import pygame
import numpy as np
import time
import threading

import rospy
from PIL import Image
from matplotlib import pyplot as plt

# 仿真环境配置
SCREEN_WIDTH = 4
SCREEN_HEIGHT = 3
ROBOT_RADIUS = 0.15
BACKGROUND_COLOR = (255, 255, 255)
ROBOT_COLOR = (0, 100, 255)
FPS = 60


class SimulationEnvironment:
    def __init__(self, num_agents=5):
        self.clock = pygame.time.Clock()
        self.num_agents = num_agents
        self.positions = np.random.rand(num_agents, 2) * np.array(
            [SCREEN_WIDTH, SCREEN_HEIGHT]
        )
        self.velocities = np.zeros((num_agents, 2))
        self.running = True
        self.trajectories = []

    def set_velocity(self, entity_id, velocity):
        self.velocities[entity_id] = np.array(velocity)
        if entity_id == 0:
            print(f"set velocity of entity {entity_id} to {velocity}")

    def get_position(self, entity_id):
        return self.positions[entity_id]

    def update_positions(self, dt):
        self.positions += self.velocities * dt
        # 边界处理
        self.positions = np.clip(
            self.positions,
            [-SCREEN_WIDTH, -SCREEN_HEIGHT],
            [SCREEN_WIDTH, SCREEN_HEIGHT],
        )
        self.trajectories.append(self.positions.copy())
        print(self.positions)

    def save_trajectories_as_gif(self, gif_filename="trajectories.gif"):
        # 创建图像列表
        images = []

        for t in range(len(self.trajectories)):
            fig, ax = plt.subplots()
            ax.set_xlim(-SCREEN_WIDTH, SCREEN_WIDTH)
            ax.set_ylim(-SCREEN_HEIGHT, SCREEN_HEIGHT)
            ax.set_aspect("equal")

            # 绘制每个机器人的位置
            for i in range(self.num_agents):
                pos = self.trajectories[t][i]
                circle = plt.Circle(
                    (pos[0], pos[1]), ROBOT_RADIUS, color=np.array(ROBOT_COLOR) / 255
                )
                ax.add_patch(circle)

            # 添加到图像列表
            plt.close(fig)  # 关闭图像以避免内存泄漏
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(Image.fromarray(image))

        # 将图像保存为GIF
        images[0].save(
            gif_filename, save_all=True, append_images=images[1:], duration=100, loop=0
        )
        print(f"Trajectories saved as {gif_filename}")


rospy.init_node("multi_robot_controller")

# 创建仿真环境
manager_sim_instance = SimulationEnvironment(num_agents=5)
