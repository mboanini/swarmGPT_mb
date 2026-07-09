"""Static AST/tokenize checks for generated motion primitives.

Verifies domain-specific structural rules that pylint cannot catch.
Run alongside pylint in the static check loop (see pipeline.py).
"""
import ast
import io
import tokenize as _tokenize


def _get_n_args_comment(source: str) -> int | None:
    """Extract N from '# n_args: N' via tokenize (ast.parse strips comments)."""
    try:
        tokens = _tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == _tokenize.COMMENT and "n_args" in tok.string:
                try:
                    return int(tok.string.split(":")[1].strip().split()[0])
                except (IndexError, ValueError):
                    return None
    except _tokenize.TokenError:
        return None
    return None


def _main_func_body(tree: ast.AST, func_name: str) -> list[ast.stmt]:
    """Body statements of the named function (restricts checks to target, ignores helpers)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.body
    return []


def _find_params_destructuring(body: list[ast.stmt]) -> list[ast.Assign]:
    """Find all `a, b, c = params` assignments in the function body."""
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
    """Find forbidden `params[i]` subscript accesses."""
    hits = []
    for stmt in body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "params"):
                hits.append(node)
    return hits


def _find_sanitize_and_drone_id_uses(body: list[ast.stmt]) -> tuple[int | None, list[int]]:
    """Return (line of _sanitize_drone_ids call, lines where drone_ids used as index)."""
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
    """Find while loops, raise, and assert statements."""
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


def _check_all_returns(body: list[ast.stmt]) -> list[str]:
    """Check 6 (partial): every return must be a 2-tuple (final_pos, waypoints)."""
    errors = []
    returns = sorted(
        [node for stmt in body for node in ast.walk(stmt) if isinstance(node, ast.Return)],
        key=lambda n: n.lineno,
    )
    if not returns:
        return ["Check 6: no return statement found"]
    for r in returns:
        if not (isinstance(r.value, ast.Tuple) and len(r.value.elts) == 2):
            errors.append(f"Check 6: return at line {r.lineno} is not a 2-tuple (final_pos, waypoints)")
    return errors


def check(source: str, func_name: str) -> list[str]:
    """Run all mechanical checks on the function source.

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

    # Check 1 + 2a: n_args comment must exist, single destructuring line, counts must match
    n_declared = _get_n_args_comment(source)
    if n_declared is None:
        failures.append("Check 1: '# n_args: N' comment is missing from the function")

    destructurings = _find_params_destructuring(body)
    if len(destructurings) != 1:
        failures.append(
            f"Check 2: expected exactly 1 'a, b, c = params' line, found {len(destructurings)}"
        )
    elif n_declared is not None:
        n_actual = len(destructurings[0].targets[0].elts)
        if n_declared != n_actual:
            failures.append(
                f"Check 1: # n_args: {n_declared} but destructuring has {n_actual} variables"
            )

    # Check 2b: no params[i] index access
    index_hits = _find_params_index_access(body)
    if index_hits:
        lines = ", ".join(str(n.lineno) for n in index_hits)
        failures.append(f"Check 2: forbidden params[i] index access at line(s) {lines}")

    # Check 3: only applies when drone_ids appears in the params destructuring
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
            failures.append("Check 3: drone_ids is in params but _sanitize_drone_ids is never called")
        else:
            early = [ln for ln in drone_index_uses if ln < sanitize_line]
            if early:
                failures.append(
                    f"Check 3: drone_ids used as index before _sanitize_drone_ids at line(s) {early}"
                )

    # Check 6 (partial): all return statements are 2-tuples
    failures.extend(_check_all_returns(body))

    # Check 7: no while / raise / assert
    forbidden = _find_forbidden_statements(body)
    if forbidden:
        failures.append(f"Check 7: forbidden statements: {'; '.join(forbidden)}")

    return failures
