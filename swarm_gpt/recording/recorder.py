"""Experiment recorder: run/command directories, JSONL events and never-overwriting file writes.

Layout (one directory per run, one per natural-language command):

    experiments/<run>/
        run.json                  created once with the run
        sessions/<ts>.json        one per application start (config, versions, launch args)
        events.jsonl              run-level events (outside of any command)
        library/                  library snapshots (see library.py)
        commands/<NNNN>_<slug>/
            command.json          written when the command starts
            events.jsonl          every stage start/end/error, appended and flushed immediately
            llm_calls.jsonl       every LLM API attempt
            result.json           written when the command is finalized (or marked interrupted)
            ...                   stage outputs (codegen/, choreography/, sim_*/, deploy_*/)

Recording must never change the behavior of the pipeline: every public method swallows and logs
its own errors, and exceptions raised by the pipeline are re-raised unchanged by the hooks.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD_ROOT = ROOT / "experiments"


def now_ns() -> dict[str, int]:
    """Wall-clock (UTC epoch) and monotonic timestamps, both in nanoseconds."""
    return {"t_wall_ns": time.time_ns(), "t_mono_ns": time.perf_counter_ns()}


def timestamp() -> str:
    """Local time as a sortable string for file names."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def to_jsonable(obj: Any) -> Any:
    """Convert numpy types, paths, sets and other objects into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if obj is Ellipsis:
        return "..."
    if hasattr(obj, "model_dump"):  # pydantic objects, e.g. OpenAI responses
        try:
            return to_jsonable(obj.model_dump(mode="json"))
        except Exception:
            pass
    return repr(obj)


def exception_info(e: BaseException) -> dict:
    """Exception type, message and full traceback; also the last attempt of a tenacity RetryError."""
    info = {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
    }
    last = getattr(e, "last_attempt", None)
    if last is not None:
        try:
            inner = last.exception()
            if inner is not None:
                info["last_attempt"] = exception_info(inner)
        except Exception:
            pass
    return info


def unique_path(path: Path) -> Path:
    """Return path, or path with a _1, _2, ... suffix if it already exists. Never overwrites."""
    if not path.exists():
        return path
    for i in range(1, 100_000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free name for {path}")


def _slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len].rstrip("_") or "command"


class Recorder:
    """Process-wide experiment recorder. Inactive (all methods are no-ops) until start_run()."""

    def __init__(self):
        self.run_dir: Path | None = None
        self.command_dir: Path | None = None
        self.command_id: str | None = None
        self.session_id: str | None = None
        self.drone_ids: dict[int, int] = {}  # Choreographer index -> Crazyflie ID
        self._lock = threading.RLock()
        self._local = threading.local()  # per-thread stage stack
        self._command_state: dict = {}
        self._counters: dict[str, int] = {}
        # Shared state of the hooks (hooks.py)
        self.pending_sim_dir: Path | None = None  # set by the simulate hook, taken by sim.py
        self.active_trace = None  # SimTrace of the running simulation
        self.last_sim_dir: Path | None = None
        self.last_splines_dir: Path | None = None  # simulation whose splines deploy() would fly
        self.active_deploy = None  # DeployRecorder during deploy()
        self.choreo_dir: Path | None = None  # choreography attempt of the last LLM answer
        self.choreo_capture: dict | None = None  # intermediate results of response2waypoints
        self.codegen: dict | None = None  # state of the primitive being generated

    # ------------------------------------------------------------------ run / command lifecycle

    @property
    def active(self) -> bool:
        return self.run_dir is not None

    def start_run(self, run_name: str | None = None, root: Path | None = None) -> Path:
        """Create or resume experiments/<run_name> (launch.py calls this only with --run)."""
        root = Path(root) if root is not None else DEFAULT_RECORD_ROOT
        self.session_id = timestamp()
        if run_name is None:
            run_name = f"unnamed_{self.session_id}"
        self.run_dir = root / run_name
        (self.run_dir / "commands").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "sessions").mkdir(exist_ok=True)
        run_file = self.run_dir / "run.json"
        if not run_file.exists():
            self._write_json_exclusive(run_file, {"run": run_name, "created": self.session_id})
        self._mark_interrupted_commands()
        self.event("session_start", session_id=self.session_id)
        return self.run_dir

    def begin_command(self, text: str) -> Path | None:
        """Finalize the open command (if any) and create a new, never-reused command directory."""
        if not self.active:
            return None
        try:
            with self._lock:
                self.end_command()
                commands = self.run_dir / "commands"
                existing = [int(p.name[:4]) for p in commands.iterdir() if p.name[:4].isdigit()]
                index = max(existing, default=0) + 1
                while True:
                    name = f"{index:04d}_{_slug(text)}"
                    try:
                        (commands / name).mkdir()
                        break
                    except FileExistsError:
                        index += 1
                self.command_dir = commands / name
                self.command_id = name
                self._counters = {}
                self._command_state = {"actions": [], "errors": []}
                self._write_json_exclusive(
                    self.command_dir / "command.json",
                    {
                        "command_id": name,
                        "text": text,
                        "session_id": self.session_id,
                        "drone_ids": self.drone_ids,
                        **now_ns(),
                    },
                )
            self.event("command_start", text=text)
            return self.command_dir
        except Exception:
            logger.exception("Recorder: could not create the command record")
            return None

    def end_command(self, reason: str = "next_command") -> None:
        """Write result.json for the open command. Called on the next command, reset or exit."""
        if self.command_dir is None:
            return
        try:
            with self._lock:
                self.event("command_end", reason=reason)
                actions = self._command_state.get("actions", [])
                last = actions[-1] if actions else None
                result = {
                    "command_id": self.command_id,
                    "finalized_by": reason,
                    "status": "failed" if last and last.get("status") == "error" else "completed",
                    "actions": actions,
                    "errors": self._command_state.get("errors", []),
                    **now_ns(),
                }
                for hook in self._command_state.get("on_end", []):
                    try:
                        result.update(hook() or {})
                    except Exception:
                        logger.exception("Recorder: end-of-command hook failed")
                self._write_json_exclusive(self.command_dir / "result.json", result)
        except Exception:
            logger.exception("Recorder: could not finalize the command record")
        finally:
            self.command_dir = None
            self.command_id = None
            self._command_state = {}

    def on_command_end(self, fn: Any) -> None:
        """Register a callable whose returned dict is merged into result.json."""
        if self.command_dir is not None:
            self._command_state.setdefault("on_end", []).append(fn)

    def record_action(self, name: str, status: str, **data: Any) -> None:
        """Summarize a top-level UI action (initial_prompt, reprompt, simulate, deploy, ...)."""
        if self.command_dir is None:
            return
        entry = {"action": name, "status": status, **to_jsonable(data), **now_ns()}
        self._command_state.setdefault("actions", []).append(entry)
        if status == "error":
            self._command_state.setdefault("errors", []).append(entry)

    def _mark_interrupted_commands(self) -> None:
        """Commands without result.json were cut off by a crash or kill: mark them, keep all data."""
        for path in sorted((self.run_dir / "commands").iterdir()):
            if path.is_dir() and not (path / "result.json").exists():
                try:
                    self._write_json_exclusive(
                        path / "result.json",
                        {"command_id": path.name, "status": "interrupted",
                         "finalized_by": f"session {self.session_id} start", **now_ns()},
                    )
                except Exception:
                    logger.exception("Recorder: could not mark %s as interrupted", path)

    # ------------------------------------------------------------------ stages and events

    @property
    def stage_stack(self) -> list[str]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @property
    def current_stage(self) -> str | None:
        return "/".join(self.stage_stack) or None

    @contextmanager
    def stage(self, name: str, **data: Any) -> Iterator[dict]:
        """Record start/end/duration/exception of a stage. Exceptions propagate unchanged."""
        if not self.active:
            yield {}
            return
        self.stage_stack.append(name)
        start = now_ns()
        out: dict = {}
        self.event("stage_start", **data)
        try:
            yield out
        except BaseException as e:
            self.event("stage_error", duration_s=self._since(start), error=exception_info(e), **out)
            raise
        else:
            self.event("stage_end", duration_s=self._since(start), **out)
        finally:
            self.stage_stack.pop()

    @staticmethod
    def _since(start: dict) -> float:
        return (time.perf_counter_ns() - start["t_mono_ns"]) / 1e9

    def event(self, kind: str, **data: Any) -> None:
        """Append one event to the command's events.jsonl (or the run's, outside a command)."""
        if not self.active:
            return
        record = {"event": kind, "stage": self.current_stage, "command_id": self.command_id,
                  **now_ns(), **data}
        self.append_jsonl("events.jsonl", record)

    def append_jsonl(self, name: str, record: dict) -> None:
        """Append a JSON line to <command or run dir>/<name>, flushed immediately."""
        if not self.active:
            return
        try:
            line = json.dumps(to_jsonable(record), ensure_ascii=False)
            with self._lock:
                base = self.command_dir or self.run_dir
                with open(base / name, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
        except Exception:
            logger.exception("Recorder: could not append to %s", name)

    # ------------------------------------------------------------------ files

    def base_dir(self) -> Path | None:
        return self.command_dir or self.run_dir

    def new_dir(self, prefix: str, parent: Path | None = None) -> Path | None:
        """Create a new numbered subdirectory <prefix>_<NNN> of the command (or run) directory."""
        if not self.active:
            return None
        try:
            with self._lock:
                parent = parent or self.base_dir()
                index = 0
                while True:
                    path = parent / f"{prefix}_{index:03d}"
                    try:
                        path.mkdir(parents=True)
                        return path
                    except FileExistsError:
                        index += 1
        except Exception:
            logger.exception("Recorder: could not create directory %s", prefix)
            return None

    def next_index(self, key: str) -> int:
        """Per-command counter, e.g. for choreography attempts."""
        with self._lock:
            value = self._counters.get(key, 0)
            self._counters[key] = value + 1
            return value

    def save_json(self, path: Path, data: Any) -> Path | None:
        """Write JSON to path (renamed with a suffix if it exists). Returns the path written."""
        try:
            path = unique_path(Path(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))
            os.replace(tmp, path)
            return path
        except Exception:
            logger.exception("Recorder: could not write %s", path)
            return None

    def save_text(self, path: Path, text: str) -> Path | None:
        try:
            path = unique_path(Path(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text if text is not None else "", encoding="utf-8")
            os.replace(tmp, path)
            return path
        except Exception:
            logger.exception("Recorder: could not write %s", path)
            return None

    def save_npz(self, path: Path, **arrays: Any) -> Path | None:
        """Write an uncompressed .npz atomically (temp file + rename), never overwriting."""
        return save_npz(path, **arrays)

    @staticmethod
    def _write_json_exclusive(path: Path, data: Any) -> None:
        """Create path; fails if it already exists (records are never silently overwritten)."""
        with open(path, "x", encoding="utf-8") as f:
            json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def save_npz(path: Path, **arrays: Any) -> Path | None:
    """Write an .npz atomically (temp file + rename), never overwriting an existing file."""
    try:
        path = unique_path(Path(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(tmp, path)
        return path
    except Exception:
        logger.exception("Recorder: could not write %s", path)
        return None


REC = Recorder()
