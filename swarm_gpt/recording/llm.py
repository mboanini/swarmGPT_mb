"""Record every LLM API attempt (request, response, usage, latency, errors) to llm_calls.jsonl.

All OpenAI calls of the pipeline (Router, Describe, Design, Write, CodeReview, Debug, test-parameter
inference, Choreographer) go through the single client object in swarm_gpt.core._llm_client, so one
wrapper around its `chat.completions.create` sees all of them, including tenacity retries. The
wrapper returns the original response object and re-raises the original exception.
"""

from __future__ import annotations

import functools
import itertools
import time
from typing import Any, Callable

from swarm_gpt.recording.recorder import REC, Recorder, exception_info, to_jsonable

_call_ids = itertools.count()


def _response_record(response: Any) -> dict:
    try:
        return to_jsonable(response.model_dump(mode="json"))
    except Exception:
        return {"repr": repr(response)}


def record_call(rec: Recorder, kind: str, request: dict, call: Callable[[], Any]) -> Any:
    """Run call(), record request/response/latency/error, return the response unchanged."""
    record = {
        "call_id": next(_call_ids),
        "api": kind,
        "stage": rec.current_stage,
        "command_id": rec.command_id,
        "request": to_jsonable(request),
        "t_start_wall_ns": time.time_ns(),
    }
    start = time.perf_counter()
    try:
        response = call()
    except BaseException as e:
        record.update(latency_s=time.perf_counter() - start, error=exception_info(e))
        rec.append_jsonl("llm_calls.jsonl", record)
        raise
    record["latency_s"] = time.perf_counter() - start
    record["response"] = _response_record(response) if kind == "openai" else to_jsonable(response)
    usage = record["response"].get("usage") if isinstance(record["response"], dict) else None
    if usage:
        record["usage"] = usage
    rec.append_jsonl("llm_calls.jsonl", record)
    return response


def install_openai_hook(client: Any, rec: Recorder = REC) -> None:
    """Wrap client.chat.completions.create in place (idempotent)."""
    completions = client.chat.completions
    original = completions.create
    if getattr(original, "_swarmgpt_recorded", False):
        return

    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        if not rec.active:
            return original(*args, **kwargs)
        return record_call(rec, "openai", {"args": args, **kwargs},
                           lambda: original(*args, **kwargs))

    create._swarmgpt_recorded = True
    completions.create = create
