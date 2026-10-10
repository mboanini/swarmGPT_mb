import ast
import re

_SHADOWED_NAMES = {"_sanitize_drone_ids", "_assign_positions", "_form_grid"}


def parse_text(text: str, lang: str = "python") -> str:
    """Extract the first code block of the given language from a markdown response.

    Raises ValueError if no matching code block is found.
    """
    pattern = rf"```{lang}.*?\s+(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        raise ValueError(f"No '{lang}' code block found in the response.")
    return matches[0].strip()


# Design docstring line declaring the params tuple: `params: tuple[...] — (name1, name2, ...)`.
# Same format primitive_writer reads to describe the primitive to the Router.
_PARAMS_LINE = re.compile(r"^\s*params\s*:\s*tuple\[.*\]\s*[—-]+\s*\((.*)\)\s*$", re.MULTILINE)


def param_names(definition: str, func_name: str) -> list[str]:
    """Return the ordered parameter names declared in the Design docstring.

    Raises ValueError if the declaration is missing or the names are not unique identifiers.
    """
    docstring = ""
    for node in ast.parse(definition).body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            docstring = ast.get_docstring(node) or ""
    match = _PARAMS_LINE.search(docstring)
    if match is None:
        raise ValueError("docstring has no 'params: tuple[...] — (name1, name2, ...)' line")
    names = [name.strip() for name in match.group(1).rstrip(", ").split(",")]
    if not all(name.isidentifier() for name in names) or len(set(names)) != len(names):
        raise ValueError(f"parameter names must be unique identifiers, got ({match.group(1)})")
    return names


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

        self._func_name = func_nodes[-1].name

        match = re.search(r"#\s*n_args:\s*(\d+)", self._code)
        self._n_args = int(match.group(1)) if match else 0

    def check_function_name(self, expected: str):
        """The expected function must be defined at module level, not nested."""
        tree = ast.parse(self._code)
        top_level = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        if expected not in top_level:
            raise ValueError(
                f"Function '{expected}' not found at module level. Found: {top_level}"
            )

    def check_n_args(self) -> None:
        """Verify the declared `# n_args: N` matches the actual arity of `params`.

        Finds the `<names> = params` destructuring assignment via ast and counts the 
        names on its left-hand side, rather than
        trusting N as self-reported by the LLM. Raises ValueError on any mismatch,
        including when no proper tuple-destructuring assignment is found at all (e.g.
        `x = params` or `x = params[0]`, which are themselves implementation bugs).
        """
        tree = ast.parse(self._code)
        unpacks = [
            n.targets[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Name) and n.value.id == "params"
            and isinstance(n.targets[0], ast.Tuple)
        ]
        if not unpacks:
            raise ValueError(
                "No `name1, name2, ... = params` destructuring assignment found — "
                "params must be unpacked with one tuple-destructuring line."
            )
        actual = len(unpacks[0].elts)
        if actual != self._n_args:
            raise ValueError(
                f"# n_args: {self._n_args} does not match the actual params arity "
                f"({actual} name(s) unpacked from params)."
            )

    def strip_shadowed_helpers(self) -> None:
        """Remove top-level redefinitions of module-level helpers that are already
        injected into every primitive's scope. Keeps any other helper functions."""
        tree = ast.parse(self._code)
        lines = self._code.splitlines()
        to_remove = [
            (n.lineno - 1, n.end_lineno)
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in _SHADOWED_NAMES
        ]
        for start, end in reversed(to_remove):
            lines[start:end] = []
        self._code = "\n".join(lines).strip()

    @property
    def function_name(self) -> str:
        return self._func_name

    @property
    def function_definition(self) -> str:
        return self._code

    @property
    def n_args(self) -> int:
        return self._n_args
