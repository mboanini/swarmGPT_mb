"""Static grammar check using pylint (errors only).

write_staging_file: shared by pipeline.py, code_review.py, debug_function.py.
check: runs pylint on the staging file and returns a list of error strings.
"""
import re
import subprocess
import sys
from pathlib import Path

from swarm_gpt.core.agent_b.parser import INJECTED_HELPERS

_ERROR_PATTERN = re.compile(r"(.*?):(\d+):(\d+): (E\w+): (.*?) \((.*?)\)")

_PREAMBLE = (
    "import numpy as np\n"
    "from numpy.typing import NDArray\n"
    f"from swarm_gpt.core.motion_primitives import {', '.join(INJECTED_HELPERS)}\n"
    "\n\n"
)
_PREAMBLE_LINES = _PREAMBLE.count("\n")  # subtracted so that pylint lines refer to the body


def write_staging_file(path: Path, body: str) -> None:
    """Write preamble + function body to the staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PREAMBLE + body + "\n")


def check(path: Path) -> list[str]:
    """Run pylint --errors-only on the staging file. Returns a list of error strings."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pylint",
            "--errors-only",
            "--msg-template={path}:{line}:{column}: {msg_id}: {msg} ({symbol})",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    errors = []
    for line in result.stdout.splitlines():
        m = _ERROR_PATTERN.match(line)
        if m:
            # group(2) = line, group(5) = message, group(6) = symbol
            errors.append(f"Line {int(m.group(2)) - _PREAMBLE_LINES}: {m.group(5)} ({m.group(6)})")
    return errors
