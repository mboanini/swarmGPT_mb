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

from modules.framework.parser import CodeParser, SingleFunctionParser, parse_text
from modules.framework.error import CodeParseError


class TestAstParser(unittest.TestCase):
    def setUp(self):
        self.code_str = """
import math
from collections import deque as d

def add(a, b=0):
    '''This function adds two numbers.'''
    return a + b

def subtract(x, y=0):
    '''This function subtracts two numbers.'''
    return x - y
        """

        self.parser = CodeParser()
        self.parser.parse_code(self.code_str)

    def test_imports(self):
        expected_imports = {"import math", "from collections import deque as d"}
        self.assertEqual(self.parser._imports, expected_imports)

    def test_function_names(self):
        expected_function_names = {"add", "subtract"}
        self.assertEqual(
            set(self.parser._function_dict.keys()), expected_function_names
        )

    def test_function_contents(self):
        expected_function_contents = [
            "def add(a, b=0):\n    '''This function adds two numbers.'''\n    return a + b",
            "def subtract(x, y=0):\n    '''This function subtracts two numbers.'''\n    return x - y",
        ]
        self.assertEqual(
            list(self.parser._function_dict.values()), expected_function_contents
        )

    def test_function_defs(self):
        expected_function_defs = {
            "add": 'def add(a, b=0):\n    """\n    This function adds two numbers.\n    """\n',
            "subtract": 'def subtract(x, y=0):\n    """\n    This function subtracts two numbers.\n    """\n',
        }
        self.assertEqual(self.parser._function_defs, expected_function_defs)


class TestTextParser(unittest.TestCase):
    def test_parse_code(self):
        text = "```python\nprint('Hello, World!')\n```"
        expected_code = "print('Hello, World!')\n"
        parsed_code = parse_text(text=text)
        self.assertEqual(parsed_code, expected_code)


class TestSingleFunctionParser(unittest.TestCase):
    def setUp(self):
        self.parser = SingleFunctionParser()

    def test_parse_code(self):
        code_str = """
def my_function(x):
    return x * x
        """
        self.parser.parse_code(code_str)
        self.assertIn("my_function", self.parser.function_names)
        self.assertIn("def my_function(x):\n", self.parser.function_defs.values())

    def test_check_function_name(self):
        code_str = """
def my_function(x):
    return x * x
        """
        self.parser.parse_code(code_str)
        self.assertRaises(
            CodeParseError, self.parser.check_function_name, "other_function"
        )
        self.parser = SingleFunctionParser()

    def test_check_error(self):
        code_str = ""  # No function in the code
        self.assertRaises(CodeParseError, self.parser.parse_code, code_str)
        code_str = """
def function1(x):
    return x * x

def function2(y):
    return y + y
        """
        self.assertRaises(CodeParseError, self.parser.parse_code, code_str)

    def test_function_definiton(self):
        code_str = """
import math
from collections import deque as d

def add(a, b=0):
    '''This function adds two numbers.'''
    return a + b
        """
        self.parser.parse_code(code_str)
        expect_res = (
            'def add(a, b=0):\n    """\n    This function adds two numbers.\n    """\n'
        )
        self.assertEqual(self.parser.function_definition, expect_res)


if __name__ == "__main__":
    unittest.main()
