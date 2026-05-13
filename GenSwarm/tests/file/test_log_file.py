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
from unittest.mock import MagicMock
from modules.file.log_file import _Logger
from modules.file.base_file import BaseFile


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = _Logger()
        self.mock_file = MagicMock(spec=BaseFile)

    def test_set_file(self):
        self.logger.set_file(self.mock_file)
        self.assertEqual(self.logger._file, self.mock_file)

    def test_log(self):
        test_content = "Test Content"
        self.logger.set_file(self.mock_file)

        # Test logging at different levels
        levels = [
            "stage",
            "action",
            "prompt",
            "response",
            "success",
            "error",
            "warning",
            "info",
            "debug",
        ]
        for level in levels:
            with self.subTest(level=level):
                self.logger.log(test_content, level)


if __name__ == "__main__":
    unittest.main()
