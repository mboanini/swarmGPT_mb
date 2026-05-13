import unittest
import numpy as np
from modules.deployment.entity.landmark import Landmark


class TestLandmark(unittest.TestCase):
    def setUp(self):
        # Initialize Landmark instance with test parameters
        initial_position = np.array([5.0, 5.0])
        self.landmark = Landmark(
            landmark_id=1, initial_position=initial_position, size=2.0, color="green"
        )

    def test_initialization(self):
        # Verify the initialization values
        self.assertEqual(self.landmark.id, 1)
        self.assertTrue(np.array_equal(self.landmark.position, np.array([5.0, 5.0])))
        self.assertEqual(self.landmark.size, 2.0)
        self.assertEqual(self.landmark.color, "green")
        self.assertFalse(self.landmark.collision)
        self.assertFalse(self.landmark.moveable)
        self.assertEqual(self.landmark.shape, "circle")
        self.assertEqual(self.landmark.state, "unvisited")

    def test_state_change(self):
        # Test changing the state of the landmark
        self.landmark.state = "visited"
        self.assertEqual(self.landmark.state, "visited")

    def tearDown(self):
        self.landmark = None


if __name__ == "__main__":
    unittest.main()
