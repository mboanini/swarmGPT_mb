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

import unittest
from unittest.mock import patch, PropertyMock, Mock
from modules.framework.constraint import ConstraintNode, ConstraintPool
from modules.file.file import File


class TestConstraintPool(unittest.TestCase):
    def setUp(self):
        self.constraint_pool = ConstraintPool()
        self.constraint_pool.reset()

    def tearDown(self):
        self.constraint_pool.reset()

    def test_initial_status(self):
        self.assertEqual(len(self.constraint_pool._constraint_nodes), 0)

    @patch("modules.file.file.logger.log")
    def test_getitem_with_existing_constraint(self, mock_logger):
        constraint_node = ConstraintNode(name="constraint1", description="desc1")
        self.constraint_pool._constraint_nodes["constraint1"] = constraint_node

        result = self.constraint_pool["constraint1"]

        self.assertEqual(result, constraint_node)
        mock_logger.assert_not_called()

    @patch("modules.file.file.logger.log")
    def test_getitem_with_non_existing_constraint(self, mock_logger):
        with self.assertRaises(Exception):
            self.constraint_pool["non_existing_constraint"]

        mock_logger.assert_called_once()

    def test_str_method(self):
        constraint_node = ConstraintNode(name="constraint1", description="desc1")
        self.constraint_pool._constraint_nodes["constraint1"] = constraint_node

        result = str(self.constraint_pool)

        self.assertEqual(result, "**constraint1**: desc1")

    def test_constaint_list_property(self):
        constraint_node = ConstraintNode(name="constraint1", description="desc1")
        self.constraint_pool._constraint_nodes["constraint1"] = constraint_node

        result = self.constraint_pool.constraint_list

        self.assertEqual(result, [{"name": "constraint1", "description": "desc1"}])

    @patch("modules.file.file.File.message", new_callable=PropertyMock)
    def test_init_constraints(self, mock_file_message):
        content = '{"constraints": [{"name": "constraint1", "description": "desc1"}]}'
        self.constraint_pool.init_constraints(content)

        self.assertTrue("constraint1" in self.constraint_pool._constraint_nodes)
        self.assertEqual(
            self.constraint_pool._constraint_nodes["constraint1"].description, "desc1"
        )
        mock_file_message.assert_called_once()

    def test_check_constraints_satisfaction_with_satisfied_constraints(self):
        constraint_node = ConstraintNode(name="constraint1", description="desc1")
        constraint_node.connect_to(Mock())
        self.constraint_pool._constraint_nodes["constraint1"] = constraint_node

        self.constraint_pool.check_constraints_satisfaction()

    @patch("modules.file.file.File.message", new_callable=Mock())
    def test_check_constraints_satisfaction_with_unsatisfied_constraints(
        self, mock_file_message
    ):
        constraint_node = ConstraintNode(name="constraint1", description="desc1")
        self.constraint_pool._constraint_nodes["constraint1"] = constraint_node

        with self.assertRaises(Exception) as context:
            self.constraint_pool.check_constraints_satisfaction()

    def test_filtered_constraints(self):
        constraint_node = ConstraintNode(name="constraint1", description="desc1")
        self.constraint_pool._constraint_nodes["constraint1"] = constraint_node

        result = self.constraint_pool.filtered_constraints([constraint_node])
        self.assertEqual(result, "**constraint1**: desc1")

        constraint_node_2 = ConstraintNode(name="constraint2", description="desc2")
        self.constraint_pool._constraint_nodes["constraint2"] = constraint_node_2
        result_2 = self.constraint_pool.filtered_constraints(
            [constraint_node, constraint_node_2]
        )

        self.assertEqual(result_2, "**constraint1**: desc1\n**constraint2**: desc2")

    @patch("modules.file.file.logger.log")
    def test_filtered_non_existed_constraints(self, mock_logger):
        constraint_node_9 = ConstraintNode(name="constraint9", description="desc2")

        with self.assertRaises(SystemExit):
            self.constraint_pool.filtered_constraints([constraint_node_9])

        mock_logger.assert_called_with(
            "Constraint constraint9 is not in the constraint pool", "error"
        )


if __name__ == "__main__":
    unittest.main()
