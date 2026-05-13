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
from unittest.mock import patch, mock_open
from modules.file.file import File, FileStatus
from modules.utils import root_manager


class TestFile(unittest.TestCase):
    def setUp(self):
        self.mock_content = "Test Content"
        self.mock_root = "/mock/root"
        self.file = File(name="test.txt", root=self.mock_root)

    @patch("modules.utils.root_manager.workspace_root", "mocked_workspace_root")
    def test_root_default_value(self):
        file = File(name="test_file.txt")
        self.assertEqual(file._root, "mocked_workspace_root")

    @patch("builtins.open", mock_open(read_data="Test Content"))
    def test_message_property(self):
        with patch.object(root_manager, "workspace_root", self.mock_root):
            message = self.file.message
            self.assertEqual(message, "Test Content")

    @patch("builtins.open", mock_open())
    def test_message_property_file_not_found(self):
        with patch.object(root_manager, "workspace_root", self.mock_root):
            message = self.file.message
            self.assertEqual(message, "")

    @patch("builtins.open", mock_open())
    def test_message_setter(self):
        with patch.object(root_manager, "workspace_root", self.mock_root):
            self.file.message = self.mock_content
            self.assertEqual(self.file._message, self.mock_content)
            self.assertEqual(self.file._status, FileStatus.NOT_TESTED)

    def test_read(self):
        with patch.object(root_manager, "workspace_root", self.mock_root):
            content = self.file.read()
            self.assertEqual(content, "File not found: /mock/root/test.txt")

    @patch("builtins.open", mock_open(read_data="Test Content"))
    def test_write(self):
        with patch.object(root_manager, "workspace_root", self.mock_root):
            self.file.write(self.mock_content)
            self.assertEqual(self.file.message, self.mock_content)

    @patch("builtins.open", mock_open())
    def test_copy(self):
        with patch.object(root_manager, "workspace_root", self.mock_root):
            new_file = self.file.copy(root=self.mock_root, name="new_test.txt")
            self.assertEqual(new_file._name, "new_test.txt")
            self.assertEqual(new_file.message, self.file.message)


if __name__ == "__main__":
    unittest.main()
