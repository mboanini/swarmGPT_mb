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

from .base_entity import Entity


class PushableObject(Entity):
    def __init__(self, object_id, initial_position, size, color="yellow"):
        super().__init__(
            object_id, initial_position, size, color=color, collision=True, movable=True
        )
        self.__target_position = None

    @property
    def target_position(self):
        """Get the target position of the robot."""
        return self.__target_position

    @target_position.setter
    def target_position(self, value):
        """Set a new target position for the robot."""
        self.__target_position = value
