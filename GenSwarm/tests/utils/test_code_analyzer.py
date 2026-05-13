import unittest
from unittest import mock
from modules.utils.code_analyzer import CodeAnalyzer  # Adjust the import as necessary


class TestCodeAnalyzer(unittest.TestCase):
    @mock.patch("modules.framework.parser.CodeParser")
    @mock.patch("radon.complexity.cc_visit")
    @mock.patch("radon.metrics.mi_visit")
    def test_analyze(self, mock_mi_visit, mock_cc_visit, mock_code_parser):
        # Mocking the CodeParser
        mock_parser = mock.Mock()
        mock_parser.function_dict = {
            "example_function": "def example_function(x): return x",
            "another_function": "def another_function(y): return y + 1",
        }
        mock_code_parser.return_value = mock_parser

        # Mocking complexity and maintainability index
        mock_cc_visit.return_value = [mock.Mock(complexity=1), mock.Mock(complexity=2)]
        mock_mi_visit.return_value = 70  # Example maintainability index

        # Sample code to analyze
        code = """
def example_function(x):
    # This is a comment
    if x > 0:
        return x
    else:
        return -x

def another_function(y):
    \"\"\"This is a docstring comment\"\"\"
    for i in range(y):
        print(i)
    """

        analyzer = CodeAnalyzer(code)
        analysis_result = analyzer.analyze()

        # Assertions
        self.assertEqual(analysis_result["average_complexity"], 1.5)
        # self.assertEqual(analysis_result['maintainability_index'], 70)
        self.assertEqual(analysis_result["total_lines"], 10)
        self.assertEqual(analysis_result["code_lines"], 8)
        self.assertEqual(analysis_result["comment_lines"], 2)
        self.assertAlmostEqual(analysis_result["comment_ratio"], 0.2, places=3)
        self.assertEqual(len(analysis_result["function_stats"]), 2)

    @mock.patch("modules.framework.parser.CodeParser")
    def test_count_lines_and_comments(self, mock_code_parser):
        mock_parser = mock.Mock()
        mock_parser.function_dict = {}
        mock_code_parser.return_value = mock_parser

        code = """
# A comment
def my_function():
    pass  # another comment
"""
        analyzer = CodeAnalyzer(code)
        (
            total_lines,
            code_lines,
            comment_lines,
            comment_ratio,
        ) = analyzer.count_lines_and_comments()

        # Assertions
        self.assertEqual(total_lines, 3)
        self.assertEqual(code_lines, 2)
        self.assertEqual(comment_lines, 1)
        self.assertAlmostEqual(comment_ratio, 1 / 3, places=3)

    @mock.patch("modules.framework.parser.CodeParser")
    def test_analyze_functions(self, mock_code_parser):
        mock_parser = mock.Mock()
        mock_parser.function_dict = {
            "func1": "def func1(): pass",
            "func2": "def func2(): pass  # comment",
        }
        mock_code_parser.return_value = mock_parser

        analyzer = CodeAnalyzer("dummy_code")
        function_stats = analyzer.analyze_functions()

        # Assertions
        self.assertEqual(len(function_stats), 2)
        self.assertIn("func1", function_stats)
        self.assertIn("func2", function_stats)


if __name__ == "__main__":
    unittest.main()
