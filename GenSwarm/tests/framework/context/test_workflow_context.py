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
import os
from unittest.mock import MagicMock, patch

from modules.framework.code import FunctionTree
from modules.framework.context import WorkflowContext
from modules.framework.constraint import ConstraintPool
from modules.file import File


class TestWorkflowContext(unittest.TestCase):
    TEST_FILE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/test_context.pkl"

    @classmethod
    def setUpClass(cls):
        cls.context = WorkflowContext()
        cls.constraint_pool = cls.context._constraint_pool
        # cls.function_pool = cls.context._function_pool

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.TEST_FILE_PATH):
            os.remove(cls.TEST_FILE_PATH)

    def test_save_and_load_constraint_pool(self):
        expected_dict = {"constraint1": "value1", "constraint2": "value2"}
        self.constraint_pool._constraint_nodes = expected_dict
        self.context.save_to_file(self.TEST_FILE_PATH)

        self.constraint_pool.reset()
        constraint_data = ConstraintPool()._constraint_nodes
        self.assertEqual(len(constraint_data), 0)

        self.context.load_from_file(self.TEST_FILE_PATH)
        constraint_data = ConstraintPool()._constraint_nodes
        self.assertEqual(constraint_data["constraint1"], "value1")
        self.assertEqual(constraint_data["constraint2"], "value2")

    # def test_save_and_load_function_pool(self):
    #     import_data = {"from apis import *", "import 1", "import 2"}
    #     # self.function_pool.import_list = import_data
    #     self.context.save_to_file(self.TEST_FILE_PATH)
    #     # self.function_pool.reset()

    #     function_pool_imports = FunctionTree("test").import_list
    #     self.assertEqual(len(function_pool_imports), 1)

    #     self.context.load_from_file(self.TEST_FILE_PATH)

    #     function_pool_imports = FunctionTree("test").import_list
    #     self.assertCountEqual(function_pool_imports, import_data)

    def test_set_root_for_files(self):
        # Mock File and FunctionTree's file attribute to avoid dependency on actual File class implementation
        mock_file_1 = MagicMock(spec=File)
        mock_file_2 = MagicMock(spec=File)
        mock_function_tree_1 = MagicMock(spec=FunctionTree)
        mock_function_tree_1.file = mock_file_2

        # Replace instance attributes with mocked instances
        with patch.object(self.context, "user_command", mock_file_1), patch.object(
            self.context, "_global_skill_tree", mock_function_tree_1
        ):
            # Run method
            root_path = "/test/root"
            self.context.set_root_for_files(root_path)

            # Assert that root attribute is set on each File and FunctionTree's file attribute
            self.assertEqual(mock_file_1.root, root_path)
            self.assertEqual(mock_file_2.root, root_path)

    def test_command_property(self):
        # Mock the user_command's message attribute
        self.context.user_command.message = "initial command"

        # Test the getter
        self.assertEqual(self.context.command, "initial command")

        # Test the setter
        self.context.command = "new command"
        self.assertEqual(self.context.user_command.message, "new command")

    def test_global_skill_tree_property(self):
        # Check that the global_skill_tree property retrieves the singleton instance's _global_skill_tree
        self.assertIsInstance(self.context.global_skill_tree, FunctionTree)

    def test_local_skill_tree_property(self):
        # Check that the local_skill_tree property retrieves the singleton instance's _local_skill_tree
        self.assertIsInstance(self.context.local_skill_tree, FunctionTree)

    def test_constraint_pool_property(self):
        # Check that the constraint_pool property retrieves the singleton instance's _constraint_pool
        self.assertIsInstance(self.context.constraint_pool, ConstraintPool)


if __name__ == "__main__":
    unittest.main()
