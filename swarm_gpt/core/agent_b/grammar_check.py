"""Static grammar check using pylint (errors only).

write_staging_file: shared by pipeline.py, code_review.py, debug_function.py.
check: runs pylint on the staging file and returns a list of error strings.
"""
import re
import subprocess
from pathlib import Path

_ERROR_PATTERN = re.compile(r"(.*?):(\d+):(\d+): (E\w+): (.*?) \((.*?)\)")

_PREAMBLE = (
    "import numpy as np\n"
    "from numpy.typing import NDArray\n"
    "from swarm_gpt.core.motion_primitives import _sanitize_drone_ids, _assign_positions, _form_grid\n"
)


def write_staging_file(path: Path, body: str) -> None:
    """Write preamble + function body to the staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PREAMBLE + "\n\n" + body + "\n")


def check(path: Path) -> list[str]:
    """Run pylint --errors-only on the staging file. Returns a list of error strings."""
    result = subprocess.run(
        [ "pylint", "--errors-only", "--msg-template={path}:{line}:{column}: {msg_id}: {msg} ({symbol})", str(path), ],
        capture_output=True,
        text=True,
        check=False,
    )
    errors = []
    for line in result.stdout.splitlines():
        m = _ERROR_PATTERN.match(line)
        if m:
            # group(2) = line, group(5) = message, group(6) = symbol
            errors.append(f"Line {m.group(2)}: {m.group(5)} ({m.group(6)})")
    return errors
