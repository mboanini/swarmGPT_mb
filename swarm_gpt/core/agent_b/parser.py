import ast
import re


def parse_text(text: str, lang: str = "python") -> str:
    """Extract the first code block of the given language from a markdown response.

    Raises ValueError if no matching code block is found.
    """
    pattern = rf"```{lang}.*?\s+(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        raise ValueError(f"No '{lang}' code block found in the response.")
    return matches[0].strip()


class FunctionParser:
    """Parse a single Python function from a code string.

    Uses ast for validation and function name extraction, but preserves
    the raw code string so that the # n_args: N comment is not lost
    (ast.unparse strips comments).
    """

    def __init__(self):
        self._code      = ""
        self._func_name = ""
        self._n_args    = 0

    def parse(self, code: str):
        self._code = code.strip()

        tree = ast.parse(self._code)
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        if not func_nodes:
            raise ValueError("No function definition found in the code.")
        if len(func_nodes) > 1:
            raise ValueError("More than one function definition found in the code.")

        self._func_name = func_nodes[0].name

        match = re.search(r"#\s*n_args:\s*(\d+)", self._code)
        self._n_args = int(match.group(1)) if match else 0

    def check_function_name(self, expected: str):
        if self._func_name != expected:
            raise ValueError(
                f"Function name mismatch: expected '{expected}', got '{self._func_name}'."
            )

    @property
    def function_name(self) -> str:
        return self._func_name

    @property
    def function_definition(self) -> str:
        return self._code

    @property
    def n_args(self) -> int:
        return self._n_args
