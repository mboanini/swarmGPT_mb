"""Offline tests for the validation of LLM-inferred runtime test parameters (no API calls)."""

import json
import os
import re
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")  # _llm_client builds its client at import

from swarm_gpt.core.agent_b import action, describe_primitive, pipeline  # noqa: E402
from swarm_gpt.core.agent_b import infer_test_params as itp  # noqa: E402
from swarm_gpt.core.agent_b import runtime_cases  # noqa: E402  (module import: pytest would collect TestParamsError)
from swarm_gpt.core.agent_b.runtime_cases import prepare_test_params, validate_test_params  # noqa: E402

DESIGN = '''```python
# n_args: 3
def form_test_line(params: tuple, swarm_pos: NDArray, tstart: float, tend: float, limits: dict[str, NDArray]) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
    """Place the selected drones on a line along x.

    Args:
        params: tuple[list[int], float, float] — (drone_ids, spacing, time_to_finish_s)
    """
    pass
```'''

BODY = '''```python
# n_args: 3
def form_test_line(params, swarm_pos, tstart, tend, limits):
    """Place the selected drones on a line along x."""
    drone_ids, spacing, time_to_finish_s = params
    drone_ids = _sanitize_drone_ids(drone_ids, swarm_pos.shape[0])
    n = len(drone_ids)
    target = np.zeros((n, 3))
    target[:, 0] = (np.arange(n) - (n - 1) / 2) * spacing
    target[:, 1] = 120.0
    target[:, 2] = 100.0
    t = min(tstart + time_to_finish_s, tend)
    final_pos = swarm_pos.copy()
    final_pos[drone_ids] = target
    return final_pos, {t: {d: target[i] for i, d in enumerate(drone_ids)}}
```'''

VALID = {1: "([1, 2], 60.0, 3.0)", 2: "([1, 2, 3, 4], 60.0, 3.0)", 3: "([2, 4, 6], 60.0, 3.0)"}
NAMES = ["drone_ids", "spacing", "time_to_finish_s"]


def scripted_inference(monkeypatch, answers: dict[int, list[str]]) -> list[str]:
    """Replace the test-parameter LLM: answers[case] are returned in order. Returns the prompts received."""
    prompts = []

    def fake_call_llm(prompt):
        prompts.append(prompt)
        answer = answers[int(re.search(r"This is scenario (\d)", prompt).group(1))].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(itp, "_call_llm", fake_call_llm)
    return prompts


def definition() -> str:
    return DESIGN.strip("`").removeprefix("python\n")


# --- validate_test_params ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_drones, drone_ids",
    [(2, [1, 2]), (4, [1, 2, 3, 4]), (6, [1, 2, 3, 4, 5, 6]), (6, [2, 5]), (4, [3]), (2, [...]), (6, [...])],
)
def test_valid_drone_ids(n_drones, drone_ids):
    assert validate_test_params(NAMES, (drone_ids, 60.0, 3.0), n_drones) == []


@pytest.mark.parametrize(
    "drone_ids",
    [[1, 2, 3, 4], [3], [1, 1], [0, 1], [-1], [], [1.0, 2.0], [True], [[1, 2]], "1, 2", 1, None],
    ids=["nonexistent", "nonexistent-single", "duplicate", "zero", "negative", "empty", "float", "bool",
         "nested", "string", "int", "none"],
)
def test_invalid_drone_ids(drone_ids):
    errors = validate_test_params(NAMES, (drone_ids, 60.0, 3.0), n_drones=2)
    assert len(errors) == 1 and errors[0].startswith("drone_ids must be")


@pytest.mark.parametrize("value, ok", [(1, True), (2, True), (3, False), (0, False), (-2, False),
                                       ("1", False), (1.0, False), (True, False)])
def test_single_drone_id(value, ok):
    assert (validate_test_params(["leader_id", "radius"], (value, 50.0), n_drones=2) == []) == ok


def test_tuple_length_must_match_interface():
    assert validate_test_params(NAMES, ([1, 2], 60.0), 2) == [
        "expected 3 values ('drone_ids', 'spacing', 'time_to_finish_s'), got 2"
    ]


def test_other_parameters_are_not_checked():
    assert validate_test_params(["steps", "width"], (10, 999.0), 2) == []


# --- prepare_test_params -------------------------------------------------------------------------


def test_prompt_lists_scenario_ids(monkeypatch):
    prompts = scripted_inference(monkeypatch, {k: [v] for k, v in VALID.items()})
    prepare_test_params(definition(), "form_test_line")
    assert "[1, 2]" in prompts[0] and "[1, 2, 3, 4]" not in prompts[0]
    assert "[1, 2, 3, 4]" in prompts[1] and "[1, 2, 3, 4, 5, 6]" in prompts[2]


def test_regenerates_once_after_invalid_answer(monkeypatch):
    prompts = scripted_inference(monkeypatch, {1: ["([1, 2, 3, 4], 60.0, 3.0)", VALID[1]], 2: [VALID[2]], 3: [VALID[3]]})
    params = prepare_test_params(definition(), "form_test_line")
    assert params == [([1, 2], 60.0, 3.0), ([1, 2, 3, 4], 60.0, 3.0), ([2, 4, 6], 60.0, 3.0)]
    assert len(prompts) == 4
    assert "rejected: drone_ids must be a non-empty list of distinct ints in 1..2" in prompts[1]


def test_two_invalid_answers_raise(monkeypatch):
    prompts = scripted_inference(monkeypatch, {1: ["([1, 2, 3, 4], 60.0, 3.0)", "([0, 1], 60.0, 3.0)"]})
    with pytest.raises(runtime_cases.TestParamsError, match=r"Case 1 \(2 drones\): drone_ids must be .* got \[0, 1\]"):
        prepare_test_params(definition(), "form_test_line")
    assert len(prompts) == 2  # no third attempt, no further scenario


def test_unparseable_answers_raise(monkeypatch):
    scripted_inference(monkeypatch, {1: ["sure! here it is", "(1, 2"]})
    with pytest.raises(runtime_cases.TestParamsError, match="not a Python tuple"):
        prepare_test_params(definition(), "form_test_line")


def test_api_error_is_not_a_test_params_error(monkeypatch):
    scripted_inference(monkeypatch, {1: [ConnectionError("API down")]})
    with pytest.raises(ConnectionError):
        prepare_test_params(definition(), "form_test_line")


# --- through the real Pipeline -------------------------------------------------------------------


class FakeClient:
    """Scripted answers for the Describe, Design, Write, CodeReview and (optionally) Debug LLM calls."""

    def __init__(self, write: str = BODY, debug: str | None = None):
        self.write, self.debug, self.prompts = write, debug, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Respond with a JSON object only" in prompt:
            content = json.dumps({"name": "form_test_line", "description": "Place the selected drones of the "
                                  "swarm on a straight line along the x axis with fixed spacing."})
        elif "## Fixed signature" in prompt:
            content = DESIGN
        elif "## Function to implement" in prompt:
            content = self.write
        elif "Please summarize the main functions" in prompt:
            content = "SATISFIED"
        elif self.debug and prompt.startswith("The following Python function has errors"):
            content = self.debug
        else:
            raise AssertionError(f"unexpected LLM call: {prompt[:80]}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def pipe(monkeypatch, tmp_path):
    """The real Pipeline with scripted LLM answers; records certification rounds and Debug calls."""
    monkeypatch.setattr(action, "client", FakeClient())
    monkeypatch.setattr(describe_primitive, "client", FakeClient())
    monkeypatch.setattr(pipeline, "_STAGING_DIR", tmp_path)
    p = pipeline.Pipeline()
    p.rounds, p.debug_calls = 0, 0
    write = pipeline.write_staging_file

    def counted_round(path, body):
        p.rounds += 1
        write(path, body)

    monkeypatch.setattr(pipeline, "write_staging_file", counted_round)
    monkeypatch.setattr(p._debugger, "run", lambda: setattr(p, "debug_calls", p.debug_calls + 1))
    return p


def test_valid_params_are_certified(pipe, monkeypatch):
    scripted_inference(monkeypatch, {k: [v] for k, v in VALID.items()})
    [node] = pipe.run(["form a test line"])
    assert node.name == "form_test_line"
    assert (pipe.rounds, pipe.debug_calls) == (1, 0)


def test_invalid_params_never_reach_certification_or_debug(pipe, monkeypatch):
    scripted_inference(monkeypatch, {1: ["([1, 2, 3, 4], 60.0, 3.0)", "([1, 2, 3, 4], 60.0, 3.0)"]})
    with pytest.raises(runtime_cases.TestParamsError):
        pipe.run(["form a test line"])
    assert (pipe.rounds, pipe.debug_calls) == (0, 0)
