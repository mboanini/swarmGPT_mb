"""Evaluate the Router's existing_system / code_generation decision on a labeled dataset.

Usage:
    python eval/router/eval_router.py [--dataset PATH] [--runs N] [--workers N] [--limit N]
                                      [--model NAME] [--base-url URL] [--api-key-env VAR] [--omit-temperature]
                                      [--delay SECONDS] [--resume RESULTS.json]

--resume completes a saved run (same dataset, catalog, model and provider): attempts that succeeded or
failed with an invalid output are reused untouched, only the API-error attempts are retried, each in
its original run and position. The result goes to a NEW --out-dir; the saved file is never modified.
--delay waits before each API request (for rate-limited providers; use with --workers 1); the wait is
not part of the measured latency.

--model replaces only the model name in the Router's request (default: the Router's own model);
JSON mode and prompts are never changed, and neither is temperature unless --omit-temperature is given
(for models that reject temperature=0; the parameter is then not sent at all and the metadata says so).
--base-url / --api-key-env select another OpenAI-compatible provider; the API key is read from the
environment and never stored.

Positive class for precision/recall/F1 is "code_generation".
Attempts that fail are never predictions: an API error (no answer from the LLM) or an invalid output
(the LLM answered but the Router could not read it) is stored with its error and counted separately.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Load router.py on its own: importing swarm_gpt.core runs its __init__, which pulls in the
# choreographer/backend (ollama, ROS, ...), none of which the router needs.
_spec = importlib.util.spec_from_file_location("router", ROOT / "swarm_gpt/core/router.py")
_router_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_router_module)
Router = _router_module.Router

PRIMITIVES_PATH = ROOT / "swarm_gpt/data/primitives.yaml"
LABELS = ("existing_system", "code_generation")


class RecordingClient:
    """Wraps the OpenAI client so the raw answer, model and tokens of each Router call can be saved.

    Router.route() returns only the decision, so this is the only way to get them without touching it.
    If `model` is given it replaces the model name in the request (evaluation only; the Router is unchanged).
    """

    def __init__(self, client, model=None, omit_temperature=False):
        self._client = client
        self.model = model
        self.omit_temperature = omit_temperature
        self.requested_model = None  # the model name actually sent: the Router's own unless overridden
        self._local = threading.local()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self.model:
            kwargs["model"] = self.model  # messages, temperature and response_format stay as the Router sent them
        if self.omit_temperature:
            kwargs.pop("temperature", None)  # only for models that reject temperature=0; nothing else is removed
        self.requested_model = kwargs["model"]
        response = self._client.chat.completions.create(**kwargs)  # an API error propagates to route_one
        usage = response.usage
        self._local.call = {
            "model": response.model,
            "temperature": kwargs.get("temperature"),
            "raw_response": response.choices[0].message.content,
            "tokens": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens} if usage else None,
        }
        return response

    def pop_call(self) -> dict:
        """The last call made by this thread (empty if it failed); forgotten afterwards."""
        call, self._local.call = getattr(self._local, "call", {}), {}
        return call


def route_one(router: Router, recorder: RecordingClient, case: dict) -> dict:
    recorder.pop_call()
    start = time.perf_counter()
    try:
        result = router.route(case["command"])
        predicted, missing, error = result["routing"], result["missing"], None
    except Exception as e:  # API / JSON errors are reported, not counted as predictions
        predicted, missing, error = None, None, repr(e)
    latency = time.perf_counter() - start
    call = recorder.pop_call()
    # no answer from the LLM -> API error; an answer the Router could not read -> invalid output
    error_type = None if error is None else ("invalid_output" if "raw_response" in call else "api_error")
    return {**case, "predicted": predicted, "missing": missing, "error": error, "error_type": error_type, "latency_s": latency, **call}


def metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r["error"] is None]
    confusion = {exp: {pred: 0 for pred in LABELS} for exp in LABELS}
    for r in valid:
        confusion[r["expected"]][r["predicted"]] += 1

    tp = confusion["code_generation"]["code_generation"]
    fp = confusion["existing_system"]["code_generation"]
    fn = confusion["code_generation"]["existing_system"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    by_category = defaultdict(lambda: [0, 0])
    for r in valid:
        by_category[r["category"]][0] += r["predicted"] == r["expected"]
        by_category[r["category"]][1] += 1

    correct = sum(r["predicted"] == r["expected"] for r in valid)
    api_errors = sum(r["error_type"] == "api_error" for r in results)
    invalid_outputs = sum(r["error_type"] == "invalid_output" for r in results)

    # end-to-end latency of route_one, over valid predictions only; failure rates are over all attempts
    latencies = [r["latency_s"] for r in valid if r.get("latency_s") is not None]
    if len(latencies) > 1:
        p95 = quantiles(latencies, n=20, method="inclusive")[18]  # 95th percentile, linear interpolation
    else:
        p95 = latencies[0] if latencies else None
    return {
        "n": len(valid),
        "errors": len(results) - len(valid),
        "api_errors": api_errors,
        "invalid_outputs": invalid_outputs,
        "api_error_rate": api_errors / len(results) if results else 0.0,
        "invalid_output_rate": invalid_outputs / len(results) if results else 0.0,
        "failure_rate": (api_errors + invalid_outputs) / len(results) if results else 0.0,
        "latency_mean_s": mean(latencies) if latencies else None,
        "latency_median_s": median(latencies) if latencies else None,
        "latency_p95_s": p95,
        "accuracy": correct / len(valid) if valid else 0.0,
        "accuracy_all_attempts": correct / len(results) if results else 0.0,  # failed attempts count as wrong
        "precision_codegen": precision,
        "recall_codegen": recall,
        "f1_codegen": f1,
        "confusion": confusion,
        "by_category": {c: {"correct": k, "n": n, "accuracy": k / n} for c, (k, n) in by_category.items()},
    }


def print_report(m: dict, runs: list[list[dict]]) -> None:
    print("\n" + "=" * 60)
    print(f"Router accuracy: {m['accuracy']:.1%}  ({m['n']} predictions, {m['errors']} errors: "
          f"{m['api_errors']} API, {m['invalid_outputs']} invalid output)")
    print(f"Accuracy on all attempts (errors count as wrong): {m['accuracy_all_attempts']:.1%}")
    print(f"Failure rates (of {m['n'] + m['errors']} attempts): API errors {m['api_error_rate']:.1%}, "
          f"invalid outputs {m['invalid_output_rate']:.1%}, total {m['failure_rate']:.1%}")
    print(f"code_generation  precision {m['precision_codegen']:.1%}  "
          f"recall {m['recall_codegen']:.1%}  F1 {m['f1_codegen']:.1%}")
    if len(runs) > 1:
        accs = [metrics(r)["accuracy"] for r in runs]
        print(f"Per-run accuracy: {', '.join(f'{a:.1%}' for a in accs)}")

    print("\nConfusion (rows = expected, cols = predicted):")
    print(f"{'':>18} {'existing':>10} {'codegen':>10}")
    for exp in LABELS:
        row = m["confusion"][exp]
        print(f"{exp:>18} {row['existing_system']:>10} {row['code_generation']:>10}")

    print("\nBy category:")
    for c, v in m["by_category"].items():
        print(f"  {c:<15} {v['accuracy']:>6.1%}  ({v['correct']}/{v['n']})")

    tokens = [r["tokens"] for run in runs for r in run if r.get("tokens")]
    if m["latency_median_s"] is None:
        print("\nLatency (valid attempts): no valid attempt")
    else:
        print(f"\nLatency (valid attempts, end-to-end): mean {m['latency_mean_s']:.2f}s, "
              f"median {m['latency_median_s']:.2f}s, p95 {m['latency_p95_s']:.2f}s")
    print(f"Tokens: prompt {sum(t['prompt'] for t in tokens)}, completion {sum(t['completion'] for t in tokens)}")

    wrong = [r for run in runs for r in run if r["error"] is None and r["predicted"] != r["expected"]]
    if wrong:
        print("\nMisrouted:")
        for r in wrong:
            print(f"  [{r['category']}] '{r['command']}' expected {r['expected']}, got {r['predicted']} "
                  f"(missing={r['missing']})")
    errors = [r for run in runs for r in run if r["error"] is not None]
    for r in errors:
        print(f"  ERROR ({r['error_type']}) '{r['command']}': {r['error']}")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resume_problem(prior: dict, dataset: list, args) -> str | None:
    """Why the saved results cannot be completed with the current settings (None if they can)."""
    meta = prior["metadata"]
    if meta["dataset_sha256"] != sha256(args.dataset) or meta["primitives_sha256"] != sha256(PRIMITIVES_PATH):
        return "the dataset or primitives.yaml changed since the saved run"
    if any([a["command"] for a in run] != [c["command"] for c in dataset] for run in prior["results"]):
        return "the saved runs do not hold the same commands, in the same order, as the dataset (--limit?)"
    if (args.model and args.model != meta["model_requested"]) or args.base_url != meta["base_url"] \
            or args.omit_temperature != meta.get("temperature_omitted", False):
        return "--model / --base-url / --omit-temperature differ from the saved run"
    return None


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset.yaml")
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole dataset N times")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="use only N commands, evenly spread over the dataset (smoke test)")
    parser.add_argument("--model", help="model to request instead of the Router's own (default: the Router's own)")
    parser.add_argument("--base-url", help="endpoint of another OpenAI-compatible provider (default: OpenAI)")
    parser.add_argument("--api-key-env", help="name of the environment variable holding the API key (default: OPENAI_API_KEY)")
    parser.add_argument("--omit-temperature", action="store_true", help="do not send temperature (for models that reject temperature=0); recorded in the metadata")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait before each API request (default: none); not part of the measured latency")
    parser.add_argument("--resume", type=Path, help="complete this saved results JSON: retry only its API-error attempts (needs a new --out-dir)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args(argv)

    with open(args.dataset) as f:
        dataset = yaml.safe_load(f)
    for case in dataset:
        assert case["expected"] in LABELS, f"bad label in {case}"
    if args.limit:
        dataset = dataset[:: max(1, len(dataset) // args.limit)][: args.limit]

    prior = None
    if args.resume:
        prior = json.loads(args.resume.read_text())
        problem = resume_problem(prior, dataset, args)
        if problem:
            sys.exit(f"Cannot resume {args.resume}: {problem}")
        if args.out_dir.resolve() == args.resume.resolve().parent:
            sys.exit("Use a new --out-dir: the saved results must not be overwritten")
        args.runs = prior["runs"]
        args.model = args.model or prior["metadata"]["model_requested"]  # retry with the model that produced the saved attempts
    # runs[i][j] = attempt of command j in run i. Resuming reuses every saved attempt except the API errors.
    runs = [[None] * len(dataset) for _ in range(args.runs)]
    for i, saved_run in enumerate(prior["results"] if prior else []):
        for j, a in enumerate(saved_run):
            if a["error_type"] != "api_error":
                runs[i][j] = {**a, "resume_status": "reused"}
    todo = [(i, j) for i, run in enumerate(runs) for j, a in enumerate(run) if a is None]
    if prior:
        print(f"Resuming {args.resume.name}: {args.runs * len(dataset) - len(todo)} attempts reused, "
              f"{len(todo)} API-error attempts to retry = {len(todo)} Router calls")
    else:
        print(f"Planned: {len(dataset)} commands x {args.runs} runs = {len(dataset) * args.runs} Router calls")
    print(f"Model requested: {args.model or 'the Router default'} | endpoint: {args.base_url or 'OpenAI default'}"
          + (" | temperature NOT sent" if args.omit_temperature else ""))

    if client is None:
        from openai import OpenAI

        options = {}
        if args.base_url:
            options["base_url"] = args.base_url
        if args.api_key_env:
            if args.api_key_env not in os.environ:
                sys.exit(f"environment variable {args.api_key_env} is not set")
            options["api_key"] = os.environ[args.api_key_env]
        client = OpenAI(**options)
    recorder = RecordingClient(client, model=args.model, omit_temperature=args.omit_temperature)
    router = Router(primitives_path=str(PRIMITIVES_PATH), openai_client=recorder)
    def attempt(ij):
        if args.delay:
            time.sleep(args.delay)  # outside route_one, so it is not part of the measured latency
        r = route_one(router, recorder, dataset[ij[1]])
        return {**r, "resume_status": "retried"} if prior else r

    with contextlib.redirect_stdout(io.StringIO()):  # the Router prints every decision
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for (i, j), r in zip(todo, pool.map(attempt, todo)):
                runs[i][j] = r

    m = metrics([r for run in runs for r in run])
    print_report(m, runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"router_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "dataset": str(args.dataset),
            "catalog": list(router.primitives),
            "runs": args.runs,
            "metadata": {
                "date": datetime.now().isoformat(timespec="seconds"),
                "model_requested": recorder.requested_model or args.model,  # the model reported by the API is kept in every attempt ("model")
                "base_url": args.base_url,
                "temperature_omitted": args.omit_temperature,  # the temperature sent in each attempt is in "temperature" (null if omitted)
                "delay_s": args.delay,
                **({"resumed_from": str(args.resume), "resumed_from_sha256": sha256(args.resume),
                    "reused_attempts": args.runs * len(dataset) - len(todo), "retried_attempts": len(todo)} if prior else {}),
                "primitives_sha256": sha256(PRIMITIVES_PATH),
                "dataset_sha256": sha256(args.dataset),
            },
            "metrics": m,
            "results": runs,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return out


if __name__ == "__main__":
    main()