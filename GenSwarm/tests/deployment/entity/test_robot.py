import unittest
import numpy as np
from modules.deployment.entity.base_entity import Entity
from modules.deployment.entity.robot import Robot


class TestRobot(unittest.TestCase):
    def setUp(self):
        """Set up test cases with default parameters."""
        self.robot_id = 1
        self.initial_position = [0.0, 0.0]
        self.size = 1.0
        self.target_position = [10.0, 10.0]
        self.color = "green"

    def test_robot_initialization(self):
        """Test initializing a Robot with default parameters."""
        robot = Robot(self.robot_id, self.initial_position, self.size)

        # Assert the Robot attributes
        self.assertEqual(robot.id, self.robot_id)
        np.testing.assert_array_equal(robot.position, np.array(self.initial_position))
        self.assertEqual(robot.size, self.size)
        self.assertEqual(robot.color, self.color)
        self.assertTrue(robot.collision)
        self.assertTrue(robot.moveable)
        self.assertIsNone(robot.target_position)

    def test_robot_initialization_with_target_position(self):
        """Test initializing a Robot with a specified target position."""
        robot = Robot(
            self.robot_id, self.initial_position, self.size, self.target_position
        )

        # Assert the Robot's target position
        np.testing.assert_array_equal(
            robot.target_position, np.array(self.target_position)
        )

    def test_set_target_position(self):
        """Test setting a new target position for the Robot."""
        robot = Robot(self.robot_id, self.initial_position, self.size)

        # Set a new target position
        new_target = [5.0, 5.0]
        robot.target_position = new_target

        # Assert the new target position
        np.testing.assert_array_equal(robot.target_position, np.array(new_target))

    def test_change_target_position(self):
        """Test changing an existing target position."""
        robot = Robot(
            self.robot_id, self.initial_position, self.size, self.target_position
        )

        # Change target position
        new_target_position = [15.0, 15.0]
        robot.target_position = new_target_position

        # Assert the updated target position
        np.testing.assert_array_equal(
            robot.target_position, np.array(new_target_position)
        )

    def test_target_position_none(self):
        """Test setting the target position to None."""
        robot = Robot(
            self.robot_id, self.initial_position, self.size, self.target_position
        )

        # Set target position to None
        robot.target_position = None

        # Assert the target position is None
        self.assertIsNone(robot.target_position)


if __name__ == "__main__":
    unittest.main()
