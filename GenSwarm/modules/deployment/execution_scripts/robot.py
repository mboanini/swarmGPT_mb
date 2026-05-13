import numpy as np
import logging
# 假设你把这些 API 放在了 api.py 中
from api import *


class Robot:
    def __init__(self):
        self._id = get_self_id()
        self.position = self.Position()
        self.omni_wheels = self.EpuckWheels()
        self.omni_proximity = self.ProximitySensor()
        self.variables = self.Variables(self._id)
        self.log = self.setup_logger()

    class Position:
        def get_position(self):
            position = get_self_position()
            return list(position)

    class EpuckWheels:
        def set_velocity(self, vx, vy):
            set_self_velocity([vx, vy])

    class ProximitySensor:
        def get_readings(self):
            env_info = get_surrounding_environment_info()
            self_id = get_self_id()

            class Reading:
                def __init__(self, value, angle):
                    self.value = value
                    self.angle = self.Angle(angle)

                class Angle:
                    def __init__(self, angle_value):
                        self._angle_value = angle_value

                    def value(self):
                        return self._angle_value

            readings = []
            for obj in env_info:
                # 避免自身，处理机器人和障碍物
                if (
                        obj['Type'] == 'obstacle' or
                        (obj['Type'] == 'robot' and obj.get('id') != self_id)
                ):
                    vec = obj['position'] - get_self_position()
                    dist = np.linalg.norm(vec)
                    angle = np.arctan2(vec[1], vec[0])  # radians
                    value = 1.0 / (dist + 1e-6)  # 越近越大
                    readings.append(Reading(value=value, angle=angle))

            return readings

    class Variables:
        def __init__(self, robot_id):
            self._store = {'id': f"00{robot_id}"}

        def get_id(self):
            return self._store['id']

        def set_attribute(self, key, value):
            self._store[key] = value

        def get_attribute(self, key):
            return self._store.get(key)

    def setup_logger(self):
        logger = logging.getLogger(f"Robot{self._id}")
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s %(name)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def stop(self):
        stop_self()

    def get_self_radius(self):
        return get_self_radius()

    def get_environment_range(self):
        return get_environment_range()

    def get_all_robots_id(self):
        return get_all_robots_id()

    def get_prey_position(self):
        return get_prey_position()

    def get_lead_position(self):
        return get_lead_position()

    def get_target_position(self):
        return get_target_position()

    def get_target_formation_points(self):
        return get_target_formation_points()

    def get_surrounding_unexplored_area(self):
        return get_surrounding_unexplored_area()

    def get_quadrant_target_position(self):
        return get_quadrant_target_position()

    def get_all_robots_initial_position(self):
        return get_all_robots_initial_position()

    def get_prey_initial_position(self):
        return get_prey_initial_position()

    def get_initial_unexplored_areas(self):
        return get_initial_unexplored_areas()
