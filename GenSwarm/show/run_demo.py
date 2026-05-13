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
import rospy
from show.robot_manager import manager_instance
from show.robot_mamager_sim import manager_sim_instance


class Robot:
    def __init__(self, robot_id, max_speed=0.2, buffer_distance=0.3):
        """
        初始化单个机器人。
        :param robot_id: 机器人的唯一标识符。
        :param max_speed: 机器人的最大速度。
        """
        self.i = None
        self.robot_id = robot_id
        self.max_speed = max_speed
        self.position = None  # 位置将在控制器中设置
        self.target_z = 0
        self.euler = None
        self.buffer_distance = buffer_distance
        self.manager = manager_instance
        # self.manager = manager_sim_instance

    def get_position(self):
        """
        获取机器人的当前位置。
        """
        self.position = self.manager.get_position(self.robot_id)
        self.euler = self.manager.get_euler(self.robot_id)
        print(f"euler of entity {self.robot_id} is {self.euler}")
        return self.position

    def update_velocity(self, velocity):
        """
        更新机器人的速度。
        :param velocity: 新的速度向量。
        """
        z_vel = self.yaw_pid_control(self.target_z)
        # z_vel = 0
        velocity = np.append(velocity, z_vel)
        self.manager.set_velocity(self.robot_id, velocity)

    def yaw_pid_control(self, target):
        kp = 1
        kd = 0.0
        dt = 0.01
        error = target - self.euler
        if error > 3.14:
            error = error - 6.28
        if error < -3.14:
            error = error + 6.28
        # print(error)
        last_error = error
        self.i = error
        z_vel = error * kp + (error - last_error) / dt * kd
        return z_vel


class MultiRobotController:
    def __init__(self, num_robots, max_speed=0.5, buffer_distance=0.2, robot_id=[]):
        """
        初始化多机器人控制器。
        :param num_robots: 机器人的数量。
        :param max_speed: 机器人的最大速度。
        :param buffer_distance: 机器人与其他机器人的最小安全距离。
        """
        self.robots = [
            Robot(robot_id=i, max_speed=max_speed, buffer_distance=buffer_distance)
            for i in robot_id
        ]
        self.num_robots = num_robots
        self.max_speed = max_speed
        self.radius = 0.15

    def normalize_velocity(self, velocity, max_speed):
        """
        归一化速度向量，确保速度不超过最大速度。
        :param velocity: 输入的速度向量。
        :param max_speed: 速度限制。
        :return: 归一化后的速度向量。
        """
        norm = np.linalg.norm(velocity)
        if norm > max_speed:
            return (velocity / norm) * max_speed
        return velocity

    def avoid_collisions(self, robot, other_robots):
        """
        计算避免与其他机器人碰撞的速度调整。
        :param robot: 当前的机器人对象。
        :param other_robots: 其他机器人的信息。
        :return: 用于避碰的速度向量。
        """
        avoid_velocity = np.array([0.0, 0.0])

        for other_robot in other_robots:
            if other_robot.robot_id == robot.robot_id:
                continue
            diff = robot.position - other_robot.position
            dist = np.linalg.norm(diff)
            if dist < robot.buffer_distance + 2 * self.radius:
                avoid_velocity += diff / dist

        return avoid_velocity

    def compute_velocity_towards_target(
        self, robot_position, target_position, max_speed
    ):
        """
        计算朝向目标的速度。
        :param robot_position: 当前机器人的位置。
        :param target_position: 目标位置。
        :param max_speed: 最大速度。
        :return: 计算后的速度向量。
        """
        direction_to_target = target_position - robot_position
        # if np.linalg.norm(direction_to_target) < 0.1:
        #     direction_to_target *= 2
        print(
            f"direction_to_target: {direction_to_target},target_position: {target_position},robot_position: {robot_position}"
        )
        return self.normalize_velocity(direction_to_target, max_speed)

    def update_velocities(self, target_positions):
        """
        更新所有机器人的位置并计算速度。
        :param target_positions: 每个机器人的目标位置。
        """
        for i, robot in enumerate(self.robots):
            target_velocity = self.compute_velocity_towards_target(
                robot.position, target_positions[i], robot.max_speed
            )
            print(f"target_velocity: {target_velocity}")
            other_robots = [r for r in self.robots if r.robot_id != robot.robot_id]
            collision_avoidance_velocity = self.avoid_collisions(robot, other_robots)
            collision_weight = 0.4
            final_velocity = (
                target_velocity + collision_avoidance_velocity * collision_weight
            )
            final_velocity = self.normalize_velocity(final_velocity, robot.max_speed)

            robot.update_velocity(final_velocity)
        # manager_sim_instance.update_positions(dt=0.1)

    def update_positions(self):
        """
        更新所有机器人的位置。
        """
        all_positions = []
        for robot in self.robots:
            position = robot.get_position()
            all_positions.append(position)
        return all_positions

    # def assign_targets(self, start_positions, target_positions):
    #     """
    #     分配目标位置，找到最佳的目标分配方案。
    #     :param start_positions: 机器人的初始位置列表。
    #     :param target_positions: 机器人的目标位置列表。
    #     :return: 最佳的目标位置列表。
    #     """
    #     from itertools import permutations
    #     import numpy as np
    #
    #     min_total_distance = float('inf')
    #     best_permutation = None
    #
    #     for perm in permutations(self.robots):
    #         total_distance = sum(
    #             np.linalg.norm(start_positions[i] - target_positions[perm[i]]) for i in range(self.num_robots))
    #         if total_distance < min_total_distance:
    #             min_total_distance = total_distance
    #             best_permutation = perm
    #
    #     # 根据最佳排列生成目标位置列表
    #     best_target_positions = [target_positions[i] for i in best_permutation]
    #
    #     return best_target_positions

    def set_positions(self, positions):
        for i, robot in enumerate(self.robots):
            manager_instance.set_position(robot.robot_id, positions[i])

    def run(self, target_positions, max_iterations=None):
        """
        持续运行多机器人控制循环，使每个机器人移动到其目标位置。
        :param max_iterations: 最大循环次数，None 表示无限循环。
        """
        rate = rospy.Rate(100)
        iteration = 0

        try:
            while not rospy.is_shutdown():
                # self.set_positions(target_positions)
                self.update_velocities(target_positions)
                rate.sleep()
                self.update_positions()

                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break
        except KeyboardInterrupt:
            print("Control loop interrupted by user.")
        finally:
            print("Saving trajectories as GIF...")
            manager_sim_instance.save_trajectories_as_gif("trajectories.gif")
            print("Trajectories saved successfully.")


# 使用示例
if __name__ == "__main__":
    robot_id = [0, 1, 2, 3, 4]
    # robot_id = [4]
    num_robots = len(robot_id)
    controller = MultiRobotController(num_robots=num_robots, robot_id=robot_id)

    # 假设有初始位置和目标位置
    start_positions = controller.update_positions()
    line_segment = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)]
    cross = [(1, -1), (1, 1), (0, 0), (1, 0), (2, 0)]
    circle = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]

    # target_positions = np.array(line_segment, dtype=np.float32)
    # target_positions = np.array(cross, dtype=np.float32)
    target_positions = np.array(circle, dtype=np.float32)
    # target_positions += np.array((0, 0.5))

    # 分配目标位置
    # best_permutation = controller.assign_targets(start_positions, target_positions)
    print(f"最佳目标位置分配: {cross}")
    # manager_instance.set_position(2, line_segment[2])
    controller.run(target_positions, max_iterations=None)
