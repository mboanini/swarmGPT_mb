"""Infer test_params for RuntimeCheck by asking the LLM to read the function interface."""
import ast
import logging
import re

from tenacity import retry, stop_after_attempt, wait_random_exponential

from swarm_gpt.core._llm_client import client

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o"

_PROMPT = """
The `params` argument of a motion primitive is described as follows:

{params_description}

Generate a valid Python tuple with realistic test values for `params`.

Rules:
- Use only plain Python literals (int, float, str, list of int) — no numpy arrays, no positions.
- This scenario has {n_drones} drones with the 1-based IDs {valid_ids}. Use no other drone IDs:
  - `drone_ids`: a non-empty list of distinct IDs taken from {valid_ids}.
  - a single-drone parameter (name ending in `_id`, e.g. `leader_id`): one ID from {valid_ids}.
- For other parameters, pick realistic values based on the description (e.g. distances in cm, angles in degrees, counts as small integers).

- The test window is ({tstart}, {tend}] seconds. Keep duration values within this window.
- This is scenario {case_number}; choose a different valid combination of values for each scenario.

Output: one Python tuple on a single line, nothing else.
Examples: (3, 150) or ({valid_ids}, 90, 'z')
{feedback}""".strip()


def _extract_params_description(function_definition: str) -> str:
    """Extract the params section from the docstring."""
    m = re.search(r"(params\s*:.*?)(?:swarm_pos\s*:|return\s*:)", function_definition, re.DOTALL)
    if m:
        return m.group(1).strip()
    return function_definition


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10))
def _call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
    )
    return response.choices[0].message.content


def infer(
    function_definition: str,
    n_drones: int,
    tstart: float,
    tend: float,
    case_index: int = 0,
    feedback: str = "",
) -> tuple | None:
    """Call the LLM to generate a test_params tuple from the function interface

    Returns a tuple on success, None if the answer cannot be parsed.
    API failures propagate (tenacity RetryError), as in every other stage.
    `feedback` explains why a previous answer was rejected.
    """
    params_description = _extract_params_description(function_definition)
    prompt = _PROMPT.format(
        params_description=params_description, n_drones=n_drones,
        valid_ids=list(range(1, n_drones + 1)),
        tstart=tstart, tend=tend, case_number=case_index + 1, feedback=feedback,
    )
    raw = _call_llm(prompt).strip()

    # Strip markdown fences if present
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    # Take the first non-empty line
    raw = next((line.strip() for line in raw.splitlines() if line.strip()), raw)

    try:
        value = ast.literal_eval(raw)
        if not isinstance(value, tuple):
            value = (value,)
        logger.info(f"InferTestParams: -> {value}")
        return value
    except (ValueError, SyntaxError) as exc:
        logger.warning(f"InferTestParams: could not parse '{raw}' ({exc})")
        return None
