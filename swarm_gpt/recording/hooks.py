"""Observation hooks around the existing pipeline. Installed once by swarm_gpt.recording.install().

Every hook calls the original function with the same arguments and returns its result (or re-raises
its exception) unchanged; recording errors are logged and never propagate. Wrapped methods keep the
original __wrapped__ target, because AppBackend.self_correct calls `self.reprompt.__wrapped__`.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from swarm_gpt.recording import library
from swarm_gpt.recording.deploy import DeployRecorder
from swarm_gpt.recording.llm import record_call
from swarm_gpt.recording.recorder import REC, exception_info, now_ns, save_npz, to_jsonable
from swarm_gpt.recording.sim_trace import VIZ_DIR_ENV, save_simulation_outputs, save_splines

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- generic wrapping


def wrap(owner: Any, name: str, make: Callable[[Callable], Callable]) -> None:
    """Replace owner.name by make(original). Works for functions, methods and staticmethods."""
    static = inspect.getattr_static(owner, name)
    if getattr(static, "_swarmgpt_recorded", False) or getattr(
        getattr(static, "__func__", None), "_swarmgpt_recorded", False
    ):
        return
    is_static = isinstance(static, staticmethod)
    original = static.__func__ if is_static else getattr(owner, name)
    wrapper = make(original)
    functools.update_wrapper(wrapper, original)
    wrapper.__wrapped__ = getattr(original, "__wrapped__", original)
    wrapper._swarmgpt_recorded = True
    setattr(owner, name, staticmethod(wrapper) if is_static else wrapper)


def safe(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a recording function; log and swallow its errors."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("Recorder hook failed in %s", getattr(fn, "__name__", fn))
        return None


@contextmanager
def action(name: str, **data: Any) -> Iterator[dict]:
    """A top-level UI action: stage events plus a summary entry in result.json."""
    start = time.perf_counter()
    with REC.stage(name, **data) as out:
        try:
            yield out
        except BaseException as e:
            REC.record_action(name, "error", duration_s=time.perf_counter() - start,
                              error=exception_info(e), **out)
            raise
        REC.record_action(name, "ok", duration_s=time.perf_counter() - start, **out)


def sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _dir(name: str) -> Path | None:
    base = REC.base_dir()
    if base is None:
        return None
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------- AppBackend


def _library_dir() -> Path:
    return REC.run_dir / "library"


def _begin_library_tracking() -> None:
    before = library.snapshot(_library_dir())
    REC.save_json(REC.command_dir / "library_before.json", library.without_texts(before))

    def after() -> dict:
        snap = library.snapshot(_library_dir())
        REC.save_json(REC.command_dir / "library_after.json", library.without_texts(snap))
        diff = library.changes(before, snap)
        REC.save_text(REC.command_dir / "library_diff.patch", diff.pop("diff"))
        REC.save_json(REC.command_dir / "library_changes.json", diff)
        return {"library_changed": diff["changed"],
                "library_hash_before": diff["library_hash_before"],
                "library_hash_after": diff["library_hash_after"],
                "functions_added": diff["functions_added"],
                "functions_modified": diff["functions_modified"]}

    REC.on_command_end(after)


def install_backend_hooks(backend_cls: Any) -> None:
    def initial_prompt(original: Callable) -> Callable:
        def wrapper(self: Any, text: str, *args: Any, **kwargs: Any) -> Any:
            if REC.active:
                REC.drone_ids = dict(getattr(self.choreographer, "agents", {}))
                if REC.begin_command(text) is not None:
                    safe(_begin_library_tracking)
                    safe(REC.save_json, REC.command_dir / "drones.json", {
                        "drone_ids": REC.drone_ids,
                        "starting_pos_m": getattr(self.choreographer, "starting_pos", {}),
                        "note": "index i (waypoint/simulation order) -> Crazyflie ID",
                    })
            with action("initial_prompt", text=text):
                return original(self, text, *args, **kwargs)

        return wrapper

    def reprompt(original: Callable) -> Callable:
        def wrapper(self: Any, message: str, *args: Any, **kwargs: Any) -> Any:
            with action("reprompt", message=message):
                return original(self, message, *args, **kwargs)

        return wrapper

    def simulate(original: Callable) -> Callable:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Iterator:
            if not REC.active:
                return (yield from original(self, *args, **kwargs))
            sim_dir = REC.new_dir("simulation")
            REC.pending_sim_dir = sim_dir
            REC.last_sim_dir = sim_dir
            start_wall = time.time()
            if self.waypoints is not None and sim_dir is not None:
                safe(save_npz, sim_dir / "input_waypoints.npz",
                     **{k: np.asarray(v) for k, v in self.waypoints.items()})
            try:
                with action("simulate", directory=str(sim_dir)):
                    sim_data = yield from original(self, *args, **kwargs)
            except BaseException as e:
                trace = REC.active_trace
                if trace is not None:
                    safe(trace.close, "error", exception_info(e))
                raise
            finally:
                REC.pending_sim_dir = None
            if sim_dir is not None and sim_data is not None:
                safe(save_simulation_outputs, sim_dir, sim_data, start_wall)
                safe(save_splines, sim_dir, self.splines, sim_data)
                REC.last_splines_dir = sim_dir
            return sim_data

        return wrapper

    def start_viz(original: Callable) -> Callable:
        def wrapper(self: Any, splines: dict, duration: float, *args: Any, **kwargs: Any) -> Any:
            if not REC.active:
                return original(self, splines, duration, *args, **kwargs)
            viz_dir = REC.new_dir("visualizer", parent=getattr(REC, "last_splines_dir", None)
                                  or getattr(REC, "last_sim_dir", None))
            if viz_dir is not None:
                os.environ[VIZ_DIR_ENV] = str(viz_dir)
            try:
                result = original(self, splines, duration, *args, **kwargs)
                REC.event("visualizer_started", directory=str(viz_dir), duration_s=duration,
                          pid=getattr(self._viz_proc, "pid", None))
                return result
            finally:
                os.environ.pop(VIZ_DIR_ENV, None)

        return wrapper

    def replay(original: Callable) -> Callable:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            with action("replay"):
                return original(self, *args, **kwargs)

        return wrapper

    def deploy(original: Callable) -> Callable:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not REC.active:
                return original(self, *args, **kwargs)
            directory = REC.new_dir("deployment")
            recorder = DeployRecorder(directory) if directory is not None else None
            meta: dict = {"args": args, "kwargs": kwargs, **now_ns()}
            safe(meta.update, _deploy_meta(self))
            clock = _ros_clock(self.drone_controller)
            if recorder is not None:
                safe(recorder.anchor, "deploy_start", clock)
                if self.drone_controller._ros_running:
                    safe(recorder.start_rosbag)
            REC.active_deploy = recorder
            try:
                with action("deploy", directory=str(directory)):
                    return original(self, *args, **kwargs)
            finally:
                REC.active_deploy = None
                if recorder is not None:
                    safe(recorder.anchor, "deploy_end", clock)
                    safe(recorder.stop_rosbag)
                    meta["t_end_wall_ns"] = time.time_ns()
                    safe(recorder.save, meta)

        return wrapper

    def reset_data(original: Callable) -> Callable:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            safe(REC.end_command, "new_command_button")
            return original(self, *args, **kwargs)

        return wrapper

    wrap(backend_cls, "initial_prompt", initial_prompt)
    wrap(backend_cls, "reprompt", reprompt)
    wrap(backend_cls, "simulate", simulate)
    wrap(backend_cls, "_start_viz", start_viz)
    wrap(backend_cls, "replay", replay)
    wrap(backend_cls, "deploy", deploy)
    wrap(backend_cls, "reset_data", reset_data)


def _deploy_meta(backend: Any) -> dict:
    meta = {
        "drone_ids_choreographer": REC.drone_ids,
        "splines_source_simulation": str(getattr(REC, "last_splines_dir", None)),
        "spline_drone_ids": list(backend.splines.keys()),
        "controller_rate_hz": backend.drone_controller.freq,
        "ros_running": backend.drone_controller._ros_running,
    }
    if backend.waypoints is not None:
        meta["trajectory_duration_s"] = float(np.max(backend.waypoints["time"][:, -1]))
    if backend.drone_controller._ros_running:
        cfs = backend.drone_controller.swarm.allcfs.crazyfliesById
        meta["crazyflies"] = {
            int(i): {"prefix": cf.prefix, "uri": getattr(cf, "uri", None),
                     "initial_position": getattr(cf, "initialPosition", None),
                     "status_at_start": getattr(cf, "status", None),
                     "pose_at_start": _pose_dict(cf)}
            for i, cf in cfs.items()
        }
        meta["ros_time_at_start_ns"] = backend.drone_controller.swarm.allcfs.get_clock().now(
        ).nanoseconds
    return meta


def _pose_dict(cf: Any) -> dict | None:
    ps = getattr(cf, "poseStamped", None)
    if not ps:
        return None
    p, q = ps["pose"].position, ps["pose"].orientation
    return {"stamp_ns": ps["timestamp_sec"] * 10**9 + ps["timestamp_nsec"],
            "pos": [p.x, p.y, p.z], "quat_xyzw": [q.x, q.y, q.z, q.w]}


# ---------------------------------------------------------------------- DroneController


def install_drone_controller_hooks(controller_cls: Any) -> None:
    def cmd_state(original: Callable) -> Callable:
        def wrapper(self: Any, crazyflie: Any, pos_ref: Any, vel_ref: Any) -> Any:
            recorder = getattr(REC, "active_deploy", None)
            if recorder is not None:
                in_spline = recorder.phases[recorder.phase] == "run_spline_trajectories"
                spline_t = getattr(self, "_spline_t", np.nan) if in_spline else np.nan
                recorder.record_command(crazyflie, pos_ref, vel_ref, spline_t)
            return original(self, crazyflie, pos_ref, vel_ref)

        return wrapper

    def phase(name: str) -> Callable:
        def make(original: Callable) -> Callable:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                recorder = getattr(REC, "active_deploy", None)
                if recorder is None:
                    return original(self, *args, **kwargs)
                previous = recorder.set_phase(name)
                recorder.anchor(f"{name}_start", _ros_clock(self))
                ros = _ros_ns(self)
                REC.event("flight_phase_start", phase=name, args=args, kwargs=kwargs, t_ros_ns=ros)
                try:
                    result = original(self, *args, **kwargs)
                except BaseException as e:
                    REC.event("flight_phase_error", phase=name, error=exception_info(e),
                              t_ros_ns=_ros_ns(self))
                    raise
                finally:
                    recorder.phase = previous
                    tstart = getattr(self, "_spline_tstart", None)
                    if name == "run_spline_trajectories" and tstart is not None:
                        recorder.spline_tstart_mono_ns = int(round(tstart * 1e9))
                REC.event("flight_phase_end", phase=name, t_ros_ns=_ros_ns(self))
                return result

            return wrapper

        return make

    wrap(controller_cls, "cmd_state", cmd_state)
    for name in ("arm", "takeoff", "move_to_initial_positions", "run_spline_trajectories",
                 "land_full_state", "land", "takeoff_low_level", "run_open_loop"):
        if hasattr(controller_cls, name):
            wrap(controller_cls, name, phase(name))


def _ros_clock(controller: Any) -> Any:
    """The ROS clock of the crazyflie_py node (stamps of the published commands), or None."""
    try:
        return controller.swarm.allcfs.get_clock if controller._ros_running else None
    except Exception:
        return None


def _ros_ns(controller: Any) -> int | None:
    try:
        return controller.swarm.allcfs.get_clock().now().nanoseconds
    except Exception:
        return None


# ---------------------------------------------------------------------- Router and Choreographer


def install_router_hooks(router_cls: Any) -> None:
    def route(original: Callable) -> Callable:
        def wrapper(self: Any, command: str, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            with REC.stage("router", command=command) as out:
                result = original(self, command, *args, **kwargs)
                out["result"] = result
            if REC.active:
                safe(REC.save_json, _dir("router") / "router.json", {
                    "input": command, "output": result,
                    "duration_s": time.perf_counter() - start,
                    "catalog": self.primitives,
                    "note": "raw LLM JSON incl. per-part reasoning is in llm_calls.jsonl "
                    "(stage .../router)",
                })
            return result

        return wrapper

    def register_primitive(original: Callable) -> Callable:
        def wrapper(self: Any, name: str, description: str, n_args: int) -> Any:
            existed = name in self.primitives
            result = original(self, name, description, n_args)
            REC.event("router_register_primitive", name=name, description=description,
                      n_args=n_args, replaced_existing=existed)
            return result

        return wrapper

    wrap(router_cls, "route", route)
    wrap(router_cls, "register_primitive", register_primitive)


def install_choreographer_hooks(choreo_cls: Any) -> None:
    def generate_choreography(original: Callable) -> Callable:
        def wrapper(self: Any, prompt: list, *args: Any, **kwargs: Any) -> Any:
            if not REC.active:
                return original(self, prompt, *args, **kwargs)
            attempt = REC.next_index("choreography")
            directory = _dir(f"choreography/attempt_{attempt:02d}")
            REC.choreo_dir = directory
            with REC.stage("choreographer_llm", attempt=attempt, model=self._model_id):
                try:
                    response = original(self, prompt, *args, **kwargs)
                finally:
                    safe(REC.save_json, directory / "messages.json", {
                        "model_id": self._model_id, "new_prompt_messages": prompt,
                        "full_conversation": self.messages})
                safe(REC.save_text, directory / "response.txt", response)
                return response

        return wrapper

    def response2waypoints(original: Callable) -> Callable:
        def wrapper(self: Any, text: str, *args: Any, **kwargs: Any) -> Any:
            if not REC.active:
                return original(self, text, *args, **kwargs)
            directory = getattr(REC, "choreo_dir", None) or _dir(
                f"choreography/attempt_{REC.next_index('choreography'):02d}")
            REC.choreo_dir = None
            capture: dict = {"input_text": text, "args": args, "kwargs": kwargs,
                             "primitive_calls": [], "sliced_steps": []}
            REC.choreo_capture = capture
            start = time.perf_counter()
            try:
                with REC.stage("waypoints"):
                    waypoints = original(self, text, *args, **kwargs)
                capture["status"] = "ok"
                safe(_save_waypoints, directory / "waypoints.npz", waypoints, self)
                return waypoints
            except BaseException as e:
                capture["status"] = "error"
                capture["error"] = exception_info(e)
                raise
            finally:
                REC.choreo_capture = None
                capture["duration_s"] = time.perf_counter() - start
                pre = capture.pop("waypoints_before_clip", None)
                if pre is not None:
                    safe(_save_waypoints, directory / "waypoints_before_clip.npz", pre, self)
                safe(REC.save_json, directory / "parse.json", capture)

        return wrapper

    def capture_result(key: str, many: bool = False) -> Callable:
        def make(original: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = original(*args, **kwargs)
                capture = getattr(REC, "choreo_capture", None)
                if capture is not None:
                    value = result
                    if key == "waypoints_before_clip":
                        value = {k: np.array(v, copy=True) for k, v in result.items()}
                    if many:
                        capture.setdefault(key, []).append(to_jsonable(value))
                    else:
                        capture[key] = value if key == "waypoints_before_clip" else to_jsonable(
                            value)
                return result

            return wrapper

        return make

    def primitive2waypoints(original: Callable) -> Callable:
        def wrapper(self: Any, fn_name: str, args: tuple, swarm_pos: Any, tstart: float,
                    tend: float) -> Any:
            pos_in = np.array(swarm_pos, copy=True)
            result = original(self, fn_name, args, swarm_pos, tstart, tend)
            capture = getattr(REC, "choreo_capture", None)
            if capture is not None:
                capture["primitive_calls"].append(to_jsonable({
                    "primitive": fn_name, "args": args, "tstart_s": tstart, "tend_s": tend,
                    "swarm_pos_in_cm": pos_in, "swarm_pos_out_cm": result[0],
                    "waypoints_cm": {str(t): v for t, v in result[1].items()},
                }))
            return result

        return wrapper

    def call_local_llama(original: Callable) -> Callable:
        def wrapper(self: Any, messages: list) -> Any:
            if not REC.active:
                return original(self, messages)
            return record_call(REC, "ollama", {"model": self._model_id, "messages": messages},
                               lambda: original(self, messages))

        return wrapper

    wrap(choreo_cls, "generate_choreography", generate_choreography)
    wrap(choreo_cls, "response2waypoints", response2waypoints)
    wrap(choreo_cls, "_slice_choreography_from_text", capture_result("sliced_steps", many=True))
    wrap(choreo_cls, "_merge_motion_primitives", capture_result("merged_steps"))
    wrap(choreo_cls, "_choreo2waypoints", capture_result("waypoints_before_clip"))
    wrap(choreo_cls, "_raw_response2waypoints", capture_result("waypoints_before_clip"))
    wrap(choreo_cls, "_primitive2waypoints", primitive2waypoints)
    wrap(choreo_cls, "_call_local_llama", call_local_llama)


def _save_waypoints(path: Path, waypoints: dict, choreographer: Any) -> None:
    n = np.asarray(waypoints["pos"]).shape[0]
    save_npz(path, **{k: np.asarray(v) for k, v in waypoints.items()},
             drone_ids=np.asarray([REC.drone_ids.get(i, -1) for i in range(n)]),
             starting_pos=np.asarray([choreographer.starting_pos[i] for i in range(n)]))


# ---------------------------------------------------------------------- Code Generator (agent_b)


def install_codegen_hooks(pipeline_module: Any, stage_classes: dict[str, Any],
                          describe_cls: Any, validator_cls: Any) -> None:
    pipeline_cls = pipeline_module.Pipeline

    def run(original: Callable) -> Callable:
        def wrapper(self: Any, user_commands: list, *args: Any, **kwargs: Any) -> Any:
            with REC.stage("code_generator", missing=user_commands) as out:
                nodes = original(self, user_commands, *args, **kwargs)
                out["certified"] = [n.name for n in nodes]
                return nodes

        return wrapper

    def run_one(original: Callable) -> Callable:
        def wrapper(self: Any, cmd: str, *args: Any, **kwargs: Any) -> Any:
            if not REC.active:
                return original(self, cmd, *args, **kwargs)
            directory = REC.new_dir("primitive", parent=_dir("codegen"))
            REC.codegen = {"dir": directory, "version": 0, "round": -1, "node": None}
            safe(REC.save_json, directory / "input.json", {"sub_command": cmd, **now_ns()})
            start = time.perf_counter()
            try:
                with REC.stage("primitive", sub_command=cmd):
                    node = original(self, cmd, *args, **kwargs)
            except BaseException as e:
                safe(_save_codegen_result, directory, "failed", REC.codegen.get("node"),
                     time.perf_counter() - start, exception_info(e))
                raise
            finally:
                REC.codegen = None
            safe(_save_codegen_result, directory, "certified", node, time.perf_counter() - start)
            return node

        return wrapper

    def describe(original: Callable) -> Callable:
        def wrapper(self: Any, user_command: str) -> Any:
            with REC.stage("describe"):
                result = original(self, user_command)
            cg = getattr(REC, "codegen", None)
            if cg:
                safe(REC.save_json, cg["dir"] / "describe.json",
                     {"name": result[0], "description": result[1]})
            return result

        return wrapper

    def validate(original: Callable) -> Callable:
        def wrapper(self: Any, name: str, desc: str) -> Any:
            result = original(self, name, desc)
            cg = getattr(REC, "codegen", None)
            if cg:
                safe(REC.save_json, cg["dir"] / "name_validation.json",
                     {"name": name, "valid": result[0], "errors": result[1],
                      "existing_names_checked": list(getattr(self, "existing_names", []))})
            return result

        return wrapper

    def stage(label: str) -> Callable:
        def make(original: Callable) -> Callable:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                cg = getattr(REC, "codegen", None)
                if cg is None:
                    return original(self, *args, **kwargs)
                node = self._node
                cg["node"] = node
                before = (node.definition, node.body)
                start = time.perf_counter()
                name = label
                if label == "debug":
                    cg["debug"] = cg.get("debug", 0) + 1
                    name = f"debug_{cg['debug']}"
                error = None
                try:
                    with REC.stage(name):
                        return original(self, *args, **kwargs)
                except BaseException as e:
                    error = exception_info(e)
                    raise
                finally:
                    safe(_save_version, cg, name, node, before, time.perf_counter() - start,
                         error, getattr(self, "_errors", None) if label == "debug" else None)

            return wrapper

        return make

    def prepare_test_params(original: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cg = getattr(REC, "codegen", None)
            with REC.stage("infer_test_params"):
                try:
                    params = original(*args, **kwargs)
                except BaseException as e:
                    if cg:
                        safe(REC.save_json, cg["dir"] / "test_params.json",
                             {"status": "error", "error": exception_info(e)})
                    raise
            if cg:
                safe(REC.save_json, cg["dir"] / "test_params.json",
                     {"status": "ok", "params": [repr(p) for p in params]})
            return params

        return wrapper

    def write_staging_file(original: Callable) -> Callable:
        def wrapper(path: Any, body: str) -> Any:
            cg = getattr(REC, "codegen", None)
            if cg:
                cg["round"] += 1
                safe(REC.save_text, cg["dir"] / f"certification_round_{cg['round']:02d}.py", body)
                safe(_round_event, cg, "start", code_sha256=sha256(body))
            return original(path, body)

        return wrapper

    def check(label: str) -> Callable:
        def make(original: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                cg = getattr(REC, "codegen", None)
                start = time.perf_counter()
                try:
                    errors = original(*args, **kwargs)
                except BaseException as e:
                    if cg:
                        safe(_round_event, cg, label, exception=exception_info(e))
                    raise
                if cg:
                    safe(_round_event, cg, label, errors=errors, passed=not errors,
                         duration_s=time.perf_counter() - start)
                return errors

            return wrapper

        return make

    wrap(pipeline_cls, "run", run)
    wrap(pipeline_cls, "_run_one", run_one)
    wrap(describe_cls, "run", describe)
    wrap(validator_cls, "validate", validate)
    for label, cls in stage_classes.items():
        wrap(cls, "run", stage(label))
    wrap(pipeline_module, "prepare_test_params", prepare_test_params)
    wrap(pipeline_module, "write_staging_file", write_staging_file)
    wrap(pipeline_module, "grammar_check", check("pylint"))
    wrap(pipeline_module, "mechanical_check", check("mechanical"))
    wrap(pipeline_module, "check_cases", check("runtime"))


def _save_version(cg: dict, stage: str, node: Any, before: tuple, duration: float,
                  error: dict | None, debug_errors: list | None) -> None:
    index = cg["version"]
    cg["version"] += 1
    code = node.definition if stage == "design" else node.body
    prefix = cg["dir"] / f"v{index:02d}_{stage}"
    REC.save_text(prefix.with_suffix(".py"), code)
    REC.save_json(prefix.with_suffix(".json"), {
        "stage": stage, "primitive": node.name, "state": str(node.state), "n_args": node.n_args,
        "code_sha256": sha256(code),
        "definition_changed": node.definition != before[0],
        "body_changed": node.body != before[1],
        "duration_s": duration, "error": error, "errors_given_to_debugger": debug_errors,
        "note": "a stage that could not parse its LLM answer keeps the previous code; the raw "
        "answer is in llm_calls.jsonl",
    })


def _round_event(cg: dict, check: str, **data: Any) -> None:
    record = {"round": cg["round"], "check": check, **now_ns(), **data}
    with open(cg["dir"] / "certification_rounds.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def _save_codegen_result(directory: Path, status: str, node: Any, duration: float,
                         error: dict | None = None) -> None:
    data = {"status": status, "duration_s": duration, "error": error}
    if node is not None:
        data.update(name=node.name, description=node.description, n_args=node.n_args,
                    definition=node.definition, code_sha256=sha256(node.body))
        name = "certified.py" if status == "certified" else "last_code.py"
        REC.save_text(directory / name, node.body)
    REC.save_json(directory / "result.json", data)
