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


class Entity:
    def __init__(
        self,
        entity_id: int,
        initial_position: list[float] | tuple | np.ndarray,
        size: list[float] | tuple | np.ndarray | float,
        color: str | tuple,
        collision: bool = False,
        movable: bool = False,
        max_speed: float = 1.0,
        mass: float = 1.0,
        density: float = 0.1,
        shape: str = "circle",
    ):
        self.__id: int = entity_id
        self.__size = size
        self.__mass: float = mass
        self.__density: float = density
        self.__shape: str = "circle" if isinstance(size, float) else "rectangle"
        self.__color: str | tuple = color
        self.__collision: bool = collision
        self.__moveable: bool = movable
        self.__force: np.ndarray = np.array([0.0, 0.0], dtype=float)
        self.__acceleration: np.ndarray = np.array([0.0, 0.0], dtype=float)
        self.__velocity: np.ndarray = np.array([0.0, 0.0], dtype=float)
        self.__position: np.ndarray = np.array(initial_position, dtype=float)
        self.__yaw: float = 0.0
        self.__max_speed: float = max_speed
        self.__angular: float = 0.0
        self.__angular_velocity: float = 0.0
        self.__angular_acceleration: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """Get the current position of the entity."""
        return self.__position

    @position.setter
    def position(self, value: list[float] | tuple | np.ndarray):
        """Set a new position for the entity."""
        if self.__moveable:
            self.__position = np.array(value, dtype=float)
        else:
            raise ValueError("Entity is not moveable.")

    @property
    def yaw(self) -> float:
        """Get the current yaw of the entity."""
        return self.__yaw

    @yaw.setter
    def yaw(self, value: float):
        """Set a new yaw for the entity."""
        self.__yaw = value

    @property
    def size(self) -> list[float] | tuple | np.ndarray | float:
        """Get the size of the entity."""
        return self.__size

    @property
    def id(self) -> int:
        """Get the unique identifier of the entity."""
        return self.__id

    @property
    def color(self) -> str | tuple:
        """Get the color of the entity."""
        return self.__color

    @property
    def shape(self) -> str:
        """Get the shape of the entity."""
        return self.__shape

    @color.setter
    def color(self, value: str | tuple):
        """Set a new color for the entity."""
        if isinstance(value, str):
            self.__color = value
        elif isinstance(value, tuple):
            self.__color = value
        else:
            raise ValueError("Color must be a string or a tuple.")

    @property
    def collision(self) -> bool:
        """Get the collision status of the entity."""
        return self.__collision

    @collision.setter
    def collision(self, value: bool):
        """Set the collision status for the entity."""
        if isinstance(value, bool):
            self.__collision = value
        else:
            raise ValueError("Collision must be a boolean.")

    @property
    def moveable(self) -> bool:
        """Get the moveable status of the entity."""
        return self.__moveable

    @moveable.setter
    def moveable(self, value: bool):
        """Set the moveable status for the entity."""
        if isinstance(value, bool):
            self.__moveable = value
        else:
            raise ValueError("Moveable must be a boolean.")

    @property
    def velocity(self) -> np.ndarray:
        """Get the current velocity of the robot."""
        return self.__velocity

    @velocity.setter
    def velocity(self, new_velocity: list[float] | tuple | np.ndarray):
        """Set a new velocity for the robot."""
        self.__velocity = np.array(new_velocity, dtype=float)

    @property
    def acceleration(self) -> np.ndarray:
        """Get the current acceleration of the robot."""
        return self.__acceleration

    @acceleration.setter
    def acceleration(self, new_acceleration: list[float] | tuple | np.ndarray):
        """Set a new acceleration for the robot."""
        self.__acceleration = np.array(new_acceleration, dtype=float)

    @property
    def max_speed(self) -> float:
        """Get the maximum speed of the leader."""
        return self.__max_speed

    @max_speed.setter
    def max_speed(self, value: float):
        """Set a new maximum speed for the leader."""
        self.__max_speed = value

    @property
    def mass(self) -> float:
        """Get the mass of the entity."""
        return self.__mass

    @mass.setter
    def mass(self, value: float):
        """Set the mass of the entity."""
        if value > 0:
            self.__mass = value
        else:
            raise ValueError("Mass must be a positive value.")

    @property
    def density(self) -> float:
        """Get the density of the entity."""
        return self.__density

    @density.setter
    def density(self, value: float):
        """Set the density of the entity."""
        if value > 0:
            self.__density = value
        else:
            raise ValueError("Density must be a positive value.")

    @property
    def force(self) -> np.ndarray:
        """Get the force applied to the entity."""
        return self.__force

    @force.setter
    def force(self, value: np.ndarray):
        """Set the force applied to the entity."""
        self.__force = value
