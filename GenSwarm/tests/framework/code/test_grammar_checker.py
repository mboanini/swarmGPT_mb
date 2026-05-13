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

import os
import unittest
from unittest.mock import patch

from modules.framework.parser import GrammarParser
from modules.utils.root import root_manager


class TestGrammarChecker(unittest.TestCase):
    def setUp(self):
        self.grammar_checker = GrammarParser()
        self.file_path = "tests/framework/code/example.py"

        self.errors = [
            {
                "file_path": self.file_path,
                "line": 3,
                "column": 11,
                "error_code": "E0602",
                "error_message": "Undefined variable 'd'",
            },
            {
                "file_path": self.file_path,
                "line": 8,
                "column": 27,
                "error_code": "E0602",
                "error_message": "Undefined variable 'global_variable'",
            },
            {
                "file_path": self.file_path,
                "line": 14,
                "column": 14,
                "error_code": "E1101",
                "error_message": "Instance of 'BadClass' has no 'mesage' member; maybe 'message'?",
            },
        ]

    @patch("modules.file.file.logger.log")
    def test_run_pylint_check(self, mock_logger):
        errors = self.grammar_checker._run_pylint_check(self.file_path)
        self.assertEqual(errors, self.errors)

    def test_find_function_name_from_error(self):
        (
            function_name,
            error_code_line,
        ) = self.grammar_checker._find_function_name_from_error(
            os.path.join(root_manager.project_root, self.file_path), 3
        )

        self.assertEqual(function_name, "my_function")
        self.assertEqual(error_code_line, "return d")

    @patch.object(GrammarParser, "_run_pylint_check")
    @patch.object(GrammarParser, "_find_function_name_from_error")
    @patch("modules.file.file.logger.log")
    def test_check_code_errors(
        self, mock_logger, mock_find_function_name, mock_run_pylint_check
    ):
        # Simulate pylint output with errors
        mock_run_pylint_check.return_value = [
            {
                "file_path": "example.py",
                "line": 10,
                "column": 5,
                "error_code": "E0602",
                "error_message": "Undefined variable 'x'",
            }
        ]

        # Simulate finding the function name in which the error occurred
        mock_find_function_name.return_value = (
            "test_function",
            "x = undefined_variable",
        )

        # Run check_code_errors
        errors = self.grammar_checker.check_code_errors("example.py")

        # Assertions
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["function_name"], "test_function")
        self.assertEqual(errors[0]["error_message"], "Undefined variable 'x'")

        # Verify that _run_pylint_check and _find_function_name_from_error were called with correct arguments
        mock_run_pylint_check.assert_called_once_with("example.py")
        mock_find_function_name.assert_called_once_with(
            file_path="example.py", error_line=10
        )


if __name__ == "__main__":
    unittest.main()
