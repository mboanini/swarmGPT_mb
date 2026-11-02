"""Static AST/tokenize checks for generated motion primitives.

Verifies domain-specific structural rules that pylint cannot catch.
Run alongside pylint in the static check loop (see pipeline.py).
The Design interface itself is validated earlier, in design_function.py;
the returned value is validated by runtime_check.py.
"""
import ast

from swarm_gpt.core.agent_b.parser import INJECTED_HELPERS, param_names


def _main_func_body(tree: ast.AST, func_name: str) -> list[ast.stmt]:
    """Body of the named function"""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.body
    return []


def _find_params_destructuring(body: list[ast.stmt]) -> list[ast.Assign]:
    """Find all `a, b, c = params` assignments in the function body"""
    matches = []
    for stmt in body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], (ast.Tuple, ast.List))
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "params"):
                matches.append(node)
    return matches


def _find_params_index_access(body: list[ast.stmt]) -> list[ast.Subscript]:
    """Find forbidden `params[i]` subscript accesses"""
    hits = []
    for stmt in body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "params"):
                hits.append(node)
    return hits


def _find_sanitize_and_drone_id_uses(body: list[ast.stmt]) -> tuple[int | None, list[int]]:
    """Return (line of _sanitize_drone_ids call, lines where drone_ids used as index)"""
    sanitize_line = None
    index_uses = []
    for stmt in body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_sanitize_drone_ids"):
                sanitize_line = node.lineno
            if isinstance(node, ast.Subscript):
                for s in ast.walk(node.slice):
                    if isinstance(s, ast.Name) and s.id == "drone_ids":
                        index_uses.append(node.lineno)
    return sanitize_line, index_uses


def _find_forbidden_statements(body: list[ast.stmt]) -> list[str]:
    """Find while loops, raise, and assert statements"""
    found = []
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.While):
                found.append(f"while loop at line {node.lineno}")
            elif isinstance(node, ast.Raise):
                found.append(f"raise at line {node.lineno}")
            elif isinstance(node, ast.Assert):
                found.append(f"assert at line {node.lineno}")
    return found


def check(source: str, func_name: str, definition: str) -> list[str]:
    """Run all mechanical checks on the function source against its Design definition.

    Returns a list of error strings; empty list means all checks passed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SyntaxError (should be caught by pylint first): {exc}"]

    body = _main_func_body(tree, func_name)
    if not body:
        return [f"Function '{func_name}' not found in source"]

    failures = []

    # exactly one `a, b, c = params` line, unpacking the Design parameters in order
    destructurings = _find_params_destructuring(body)
    if len(destructurings) != 1:
        failures.append(
            f"Check: expected exactly one 'a, b, c = params' line, found {len(destructurings)}"
        )
    else:
        unpacked = [ast.unparse(elt) for elt in destructurings[0].targets[0].elts]
        try:
            expected = param_names(definition, func_name)
        except ValueError as exc:
            failures.append(f"Check: cannot read the Design parameter declaration: {exc}")
        else:
            if unpacked != expected:
                failures.append(
                    f"Check: params must be unpacked as '{', '.join(expected)} = params' "
                    f"(Design order), got '{', '.join(unpacked)} = params'"
                )

    # no params[i] index access
    index_hits = _find_params_index_access(body)
    if index_hits:
        lines = ", ".join(str(n.lineno) for n in index_hits)
        failures.append(f"Check: forbidden params[i] index access at line(s) {lines}")

    # only applies when drone_ids appears in the params destructuring
    has_drone_ids = (
        len(destructurings) == 1
        and any(
            isinstance(elt, ast.Name) and elt.id == "drone_ids"
            for elt in destructurings[0].targets[0].elts
        )
    )
    if has_drone_ids:
        sanitize_line, drone_index_uses = _find_sanitize_and_drone_id_uses(body)
        if sanitize_line is None:
            failures.append("Check: drone_ids is in params but _sanitize_drone_ids is never called")
        else:
            early = [ln for ln in drone_index_uses if ln < sanitize_line]
            if early:
                failures.append(
                    f"Check: drone_ids used as index before _sanitize_drone_ids at line(s) {early}"
                )

    # no while / raise / assert
    forbidden = _find_forbidden_statements(body)
    if forbidden:
        failures.append(f"Check: forbidden statements: {'; '.join(forbidden)}")

    # injected helpers are already in scope: a definition (top-level or nested) would shadow them
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in sorted(defined & set(INJECTED_HELPERS)):
        failures.append(
            f"Check: '{name}' is an injected helper already in scope: call it, do not define it"
        )

    return failures
