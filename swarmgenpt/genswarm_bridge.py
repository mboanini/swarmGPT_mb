"""GenSwarm Bridge: LLM-based synthesiser for new Crazyflie motion primitives.

Replicates the GenSwarm code-generation pipeline adapted to the swarmGPT
motion-primitive interface.  The LLM receives a detailed system prompt that
describes the exact function signature, physical constraints and available
helper utilities, then returns a self-contained Python function.

The generated code is validated with an AST safety pass before it is handed
back to the caller (DynamicLibrary.register).
"""

from __future__ import annotations

import ast
import logging
import os
import re
import textwrap
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt injected into every synthesis call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert drone-swarm choreographer writing Python motion primitives
    for Bitcraze Crazyflie drones controlled via the swarmGPT library.

    ## Motion primitive contract

    Every primitive MUST have this exact signature:

        def <fn_name>(
            params: tuple,
            swarm_pos: "NDArray",   # shape (n_drones, 3), coordinates in **cm**
            tstart: float,          # start time (seconds)
            tend: float,            # end time (seconds)
            limits: dict,           # {"lower": NDArray[3], "upper": NDArray[3]} in **meters**
        ) -> tuple["NDArray", dict[float, dict[int, "NDArray"]]]:

    Returns:
    - final_swarm_pos  : NDArray (n_drones, 3) in cm — position after the move
    - waypoints        : {timestamp_s: {drone_id: NDArray([x, y, z])}} in cm
                         At least one timestamp entry (tend) must be present.

    ## Physical limits (hard constraints — never violate)
    - Coordinates      : limits["lower"] * 100  ≤  pos  ≤  limits["upper"] * 100  (in cm)
    - Min drone spacing: 60 cm at ALL waypoints (use Hungarian assignment for collision-free paths)
    - Max velocity     : 100 cm/s  →  |Δpos| / Δt ≤ 100 cm/s per step

    ## Available imports (already in scope — do NOT import anything else)
        import numpy as np
        from numpy.typing import NDArray
        from scipy.optimize import linear_sum_assignment   # for _assign_positions
        from scipy.spatial.transform import Rotation as R

    ## Helper functions you may call (already imported at module level)
        _assign_positions(pos: NDArray, des_pos: NDArray) -> NDArray
            Hungarian optimal assignment — returns index array.
        _form_grid(swarm_pos, limits, height=None, spacing=None) -> NDArray
            Arrange drones in a tight rectangular grid.

    ## Output format

    Return ONLY a fenced Python code block — no prose before or after.
    The block must contain:
      1. A module-level docstring (one line) for the function.
      2. The complete function definition with the params tuple destructured
         in the first line of the body and a comment listing each param name.
      3. A PARAMS_SCHEMA comment directly above the def:
             # PARAMS_SCHEMA: n_args=<N>
         where N equals the number of elements in the params tuple.

    Example skeleton:

    ```python
    # PARAMS_SCHEMA: n_args=3
    def my_formation(
        params: tuple,
        swarm_pos: NDArray,
        tstart: float,
        tend: float,
        limits: dict,
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        \"\"\"Move drones into my_formation shape.\"\"\"
        steps, height, radius = params  # steps: int, height: cm, radius: cm

        n_drones = swarm_pos.shape[0]
        lim_lower, lim_upper = limits["lower"], limits["upper"]
        # ... implementation ...
        waypoints = {}
        waypoints[tend] = {i: p.copy() for i, p in enumerate(swarm_pos)}
        return swarm_pos, waypoints
    ```
""")

_USER_TEMPLATE = textwrap.dedent("""\
    ## Task
    Generate a motion primitive named `{fn_name}` that implements the following
    behaviour for {n_drones} Crazyflie drones:

    {description}

    ## Spatial envelope
    - Lower limit (m): {lim_lower}
    - Upper limit (m): {lim_upper}

    ## Requirements
    - Produce smooth, collision-free waypoints every second (steps = int(tend - tstart)).
    - Enforce min 60 cm spacing between drones at every waypoint.
    - Clip all z values to [limits["lower"][2]*100, limits["upper"][2]*100].
    - The function name must be exactly `{fn_name}`.
    - Choose the minimal set of params needed; document their names in the
      destructuring comment.

    Return ONLY the fenced ```python ... ``` block.
""")

# ---------------------------------------------------------------------------
# Banned AST node types for safety
# ---------------------------------------------------------------------------
_BANNED_NODES = {
    ast.Import,       # no extra imports allowed
    ast.ImportFrom,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
}
_BANNED_CALLS = {
    "eval", "exec", "compile", "open", "input",
    "__import__", "breakpoint", "exit", "quit",
}


@dataclass
class SynthesisResult:
    fn_name: str
    source_code: str
    n_args: int
    description: str


class GenSwarmBridge:
    """Synthesises new Crazyflie motion primitives using an LLM backend.

    Supports both the Anthropic (Claude) and OpenAI APIs.  Set the environment
    variable ANTHROPIC_API_KEY to use Claude (recommended); fall back to
    OPENAI_API_KEY / OpenAI otherwise.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self._client, self._backend, self._model_id = self._init_client(model_id)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def synthesize(
        self,
        fn_name: str,
        description: str,
        n_drones: int,
        limits: dict,
    ) -> SynthesisResult:
        """Ask the LLM to generate a new motion primitive.

        Args:
            fn_name: Python identifier for the new function.
            description: Natural-language behaviour description.
            n_drones: Hint for the number of drones (context only).
            limits: {"lower": NDArray[3], "upper": NDArray[3]} in meters.

        Returns:
            SynthesisResult with validated source code.
        """
        if not fn_name.isidentifier():
            raise ValueError(f"'{fn_name}' is not a valid Python identifier")

        user_prompt = _USER_TEMPLATE.format(
            fn_name=fn_name,
            description=description,
            n_drones=n_drones,
            lim_lower=list(limits["lower"]),
            lim_upper=list(limits["upper"]),
        )

        logger.info("Synthesising primitive '%s' via %s/%s", fn_name, self._backend, self._model_id)
        raw_response = self._call_llm(user_prompt)

        source_code, n_args = self._extract_and_validate(raw_response, fn_name)
        return SynthesisResult(
            fn_name=fn_name,
            source_code=source_code,
            n_args=n_args,
            description=description,
        )

    # ------------------------------------------------------------------
    # LLM backend
    # ------------------------------------------------------------------

    @staticmethod
    def _init_client(model_id: str | None):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            mid = model_id or "claude-sonnet-4-6"
            return client, "anthropic", mid

        if openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            mid = model_id or "gpt-4o"
            return client, "openai", mid

        raise EnvironmentError(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. "
            "Export one of them to use GenSwarmBridge."
        )

    def _call_llm(self, user_prompt: str) -> str:
        if self._backend == "anthropic":
            response = self._client.messages.create(
                model=self._model_id,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        # OpenAI
        response = self._client.chat.completions.create(
            model=self._model_id,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Extraction & validation
    # ------------------------------------------------------------------

    def _extract_and_validate(self, raw: str, expected_fn_name: str) -> tuple[str, int]:
        """Pull the code block, check safety, return (source_code, n_args)."""
        source_code = self._extract_code_block(raw)
        n_args = self._extract_n_args(source_code)
        self._validate_ast(source_code, expected_fn_name)
        return source_code, n_args

    @staticmethod
    def _extract_code_block(text: str) -> str:
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if not match:
            # Allow plain code without fences as last resort
            if "def " in text:
                return text.strip()
            raise ValueError(
                "LLM response did not contain a ```python ... ``` code block."
            )
        return match.group(1).strip()

    @staticmethod
    def _extract_n_args(source_code: str) -> int:
        match = re.search(r"#\s*PARAMS_SCHEMA\s*:\s*n_args\s*=\s*(\d+)", source_code)
        if not match:
            raise ValueError(
                "Generated code is missing the required "
                "# PARAMS_SCHEMA: n_args=<N> comment."
            )
        return int(match.group(1))

    @staticmethod
    def _validate_ast(source_code: str, expected_fn_name: str) -> None:
        """Raise ValueError if the code contains unsafe constructs."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            raise ValueError(f"Generated code has syntax errors: {exc}") from exc

        # Check for banned node types at top level (imports, etc.)
        for node in ast.walk(tree):
            if type(node) in _BANNED_NODES:
                raise ValueError(
                    f"Generated code contains banned construct: {type(node).__name__}"
                )

        # Check for banned built-in calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (
                    fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute)
                    else None
                )
                if name in _BANNED_CALLS:
                    raise ValueError(
                        f"Generated code calls banned function: '{name}'"
                    )

        # Ensure the expected function name is defined
        defined = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        ]
        if expected_fn_name not in defined:
            raise ValueError(
                f"Generated code does not define a function named '{expected_fn_name}'. "
                f"Found: {defined}"
            )
