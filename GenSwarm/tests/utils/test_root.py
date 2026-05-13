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
from modules.utils.root import root_manager, get_project_root
from pathlib import Path


class TestRootManager(unittest.TestCase):
    def setUp(self):
        self.root_manager = root_manager

    def test_get_project_root(self):
        # TODO: get path by code
        supposed_dir = Path("/catkin_ws/src/code_llm")
        actual_dir = get_project_root()
        self.assertEqual(
            actual_dir, supposed_dir, f"Expected {supposed_dir}, got {actual_dir}"
        )

    def test_update_root_with_workspace_root(self):
        # Mock the provided workspace root
        fake_workspace_root = Path("/fake_workspace_root")
        self.root_manager.update_root(workspace_root=fake_workspace_root)
        self.assertEqual(self.root_manager.workspace_root, fake_workspace_root)
        self.assertEqual(self.root_manager.data_root, fake_workspace_root / "data")


if __name__ == "__main__":
    unittest.main()
