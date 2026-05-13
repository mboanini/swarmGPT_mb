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
from modules.framework.code import FunctionNode, State
from modules.framework.constraint import ConstraintNode


class TestNodes(unittest.TestCase):
    def setUp(self):
        self.constraint_node = ConstraintNode("Constraint1", "Description1")
        self.function_node1 = FunctionNode("Function1", "Description1")
        self.function_node2 = FunctionNode("Function2", "Description2")

    def test_constraint_node_properties(self):
        self.assertEqual(self.constraint_node.name, "Constraint1")
        self.assertEqual(self.constraint_node.description, "Description1")
        self.assertEqual(self.constraint_node.brief, "**Constraint1**: Description1")

    def test_function_node_properties(self):
        self.assertEqual(self.function_node1.name, "Function1")
        self.assertEqual(self.function_node1.description, "Description1")
        self.assertEqual(self.function_node1.brief, "**Function1**: Description1")

    def test_connect_nodes(self):
        self.constraint_node.connect_to(self.function_node1)
        self.assertIn(self.constraint_node, self.function_node1.connections)
        self.assertIn(self.function_node1, self.constraint_node.connections)

    def test_has_no_connections(self):
        self.assertTrue(self.constraint_node.has_no_connections())
        self.function_node1.connect_to(self.function_node2)
        self.assertFalse(self.function_node1.has_no_connections())

    def test_add_import(self):
        imports = {"import module1", "from module2 import function"}
        self.function_node1.add_import(imports)
        self.assertListEqual(list(self.function_node1._import_list), list(imports))

    def test_add_callee_and_caller(self):
        self.function_node1.add_callee(self.function_node2)
        self.assertIn(self.function_node2, self.function_node1.callees)
        self.assertIn(self.function_node1, self.function_node2.callers)

    def test_function_body(self):
        self.function_node1.add_import({"import module1"})
        self.function_node1.content = "def test_function():\n    pass"
        expected_body = "import module1\ndef test_function():\n    pass"
        self.assertEqual(self.function_node1.function_body, expected_body)

    def test_body(self):
        function_node = FunctionNode("function1", "descption1")
        function_node._definition = "def function1():"
        self.assertEqual(function_node.body, "def function1():")

    def test_function_node_state(self):
        self.assertEqual(self.function_node1.state, State.NOT_STARTED)

        self.function_node1.state = State.DESIGNED
        self.assertEqual(self.function_node1.state, State.DESIGNED)

        self.function_node1.state = State.WRITTEN
        self.assertEqual(self.function_node1.state, State.WRITTEN)

        self.function_node1.state = 3
        self.assertEqual(self.function_node1.state, State.REVIEWED)

        # 测试设置无效状态
        with self.assertRaises(ValueError):
            self.function_node1.state = 9

    def test_function_node_reset(self):
        function_node = FunctionNode("function1", "descption1")
        function_node.reset()

        self.assertEqual(len(function_node._import_list), 0)
        self.assertEqual(function_node.content, "")
        self.assertEqual(function_node._definition, "")
        self.assertEqual(function_node.state, State.NOT_STARTED)


if __name__ == "__main__":
    unittest.main()
