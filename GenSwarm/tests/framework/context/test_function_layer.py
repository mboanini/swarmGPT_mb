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

from modules.framework.code import FunctionLayer, FunctionNode


import unittest


class TestFunctionLayer(unittest.TestCase):
    def setUp(self):
        self.function1 = FunctionNode(name="func1", description="Function 1")
        self.function2 = FunctionNode(name="func2", description="Function 2")
        self.function3 = FunctionNode(name="func3", description="Function 3")
        self.function_layer = FunctionLayer({self.function1, self.function2})

    def test_add_function(self):
        self.function_layer.add_function(self.function3)
        self.assertIn(self.function3, self.function_layer._layer)

    def test_functions_property(self):
        self.assertCountEqual(
            self.function_layer.functions, [self.function1, self.function2]
        )

    def test_next_function(self):
        with self.assertRaises(StopIteration):
            for _ in range(len(self.function_layer) + 1):
                next(self.function_layer)

    def test_empty_layer(self):
        function_layer = FunctionLayer()
        self.assertEqual(len(function_layer), 0)
