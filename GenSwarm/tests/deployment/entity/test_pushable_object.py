import unittest
import numpy as np
from modules.deployment.entity.pushable_object import PushableObject


class TestPushableObject(unittest.TestCase):
    def setUp(self):
        # Initialize PushableObject instance with test parameters
        initial_position = np.array([0.0, 0.0])
        self.pushable_object = PushableObject(
            object_id=1, initial_position=initial_position, size=1.0
        )

    def test_initialization(self):
        # Verify the initialization values
        self.assertEqual(self.pushable_object.id, 1)
        self.assertTrue(
            np.array_equal(self.pushable_object.position, np.array([0.0, 0.0]))
        )
        self.assertEqual(self.pushable_object.size, 1.0)
        self.assertEqual(self.pushable_object.color, "yellow")
        self.assertTrue(self.pushable_object.collision)
        self.assertTrue(self.pushable_object.moveable)

    def test_target_position_getter_and_setter(self):
        # Test setting and getting target_position
        new_target_position = np.array([2.0, 3.0])
        self.pushable_object.target_position = new_target_position
        self.assertTrue(
            np.array_equal(self.pushable_object.target_position, new_target_position)
        )

    def test_default_target_position(self):
        # Test default value of target_position
        self.assertIsNone(self.pushable_object.target_position)

    def tearDown(self):
        self.pushable_object = None


if __name__ == "__main__":
    unittest.main()
