import unittest
import numpy as np
from modules.deployment.utils.quad_tree import QuadTree
from modules.deployment.engine.base_engine import Entity


class MockEntity(Entity):
    """
    MockEntity to simulate a more complex entity with additional properties.
    """

    def __init__(
        self,
        entity_id: int,
        initial_position: list[float] | tuple | np.ndarray = np.zeros(2),
        size: list[float] | tuple | np.ndarray | float = 1.0,
        color: str | tuple = "blue",
        collision: bool = False,
        movable: bool = False,
        max_speed: float = 1.0,
        mass: float = 1.0,
        density: float = 0.1,
        shape: str = "circle",
    ):
        super().__init__(
            entity_id,
            initial_position,
            size,
            color,
            collision,
            movable,
            max_speed,
            mass,
            density,
            shape,
        )


class TestQuadTree(unittest.TestCase):
    def setUp(self):
        """
        Set up a QuadTree instance before each test.
        """
        self.quadtree = QuadTree(0, 0, 100, 100)

    def test_insert(self):
        """
        Test that entities are inserted into the QuadTree.
        """
        entity = MockEntity(1, initial_position=(10, 10), size=5, movable=True)
        self.quadtree.insert(entity)
        self.assertIn(entity, self.quadtree.entities)

    def test_split(self):
        """
        Test that the QuadTree splits when the maximum entities limit is exceeded.
        """
        entities = [
            MockEntity(i, initial_position=(x * 10, y * 10), movable=True, size=5)
            for i, (x, y) in enumerate(zip(range(5), range(5)))
        ]
        for entity in entities:
            self.quadtree.insert(entity)
        self.assertTrue(self.quadtree.nodes)  # QuadTree should split
        self.assertEqual(len(self.quadtree.nodes), 4)  # Should have four subnodes

    def test_retrieve(self):
        """
        Test that the QuadTree retrieves relevant entities.
        """
        entity1 = MockEntity(1, initial_position=(10, 10), size=5, movable=True)
        entity2 = MockEntity(2, initial_position=(50, 50), size=5, movable=True)
        self.quadtree.insert(entity1)
        # self.quadtree.insert(entity2)

        # Retrieve entity within bounds
        retrieved_entities = self.quadtree.retrieve(
            MockEntity(3, initial_position=(10, 10), size=5, movable=True)
        )
        self.assertIn(entity1, retrieved_entities)
        self.assertNotIn(entity2, retrieved_entities)

    def test_clear(self):
        """
        Test that the QuadTree is cleared of all entities and nodes.
        """
        entity = MockEntity(1, initial_position=(10, 10), size=5, movable=True)
        self.quadtree.insert(entity)
        self.quadtree.clear()
        self.assertEqual(len(self.quadtree.entities), 0)
        self.assertEqual(len(self.quadtree.nodes), 0)

    def test_remove(self):
        """
        Test that an entity can be removed from the QuadTree.
        """
        entity = MockEntity(1, initial_position=(10, 10), size=5, movable=True)
        self.quadtree.insert(entity)
        self.quadtree.remove(entity)
        self.assertNotIn(entity, self.quadtree.entities)

    def test_update(self):
        """
        Test that an entity's position can be updated within the QuadTree.
        """
        entity = MockEntity(1, initial_position=(10, 10), size=5, movable=True)
        self.quadtree.insert(entity)
        # Update entity's position
        entity.position = (60, 60)
        self.quadtree.update(entity)

        # Check if entity has been moved to appropriate node
        retrieved_entities = self.quadtree.retrieve(
            MockEntity(2, initial_position=(60, 60), size=5, movable=True)
        )
        self.assertIn(entity, retrieved_entities)

    def test_get_index(self):
        """
        Test the get_index method for correct quadrant identification.
        """
        entity = MockEntity(1, initial_position=(10, 10), size=5, movable=True)
        index = self.quadtree.get_index(entity)
        self.assertEqual(index, 0)  # Expected to be in the top-left quadrant

        entity = MockEntity(2, initial_position=(75, 75), size=5, movable=True)
        index = self.quadtree.get_index(entity)
        self.assertEqual(index, 3)  # Expected to be in the bottom-right quadrant


if __name__ == "__main__":
    unittest.main()
