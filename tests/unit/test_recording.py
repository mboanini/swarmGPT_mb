"""Offline tests for the experiment recording (no LLM API calls, no real simulation or ROS)."""

import inspect
import json
import os
import shutil
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")  # _llm_client builds its client at import

from swarm_gpt.recording import hooks, library  # noqa: E402
from swarm_gpt.recording.llm import install_openai_hook  # noqa: E402
from swarm_gpt.recording.recorder import REC, Recorder  # noqa: E402
from swarm_gpt.recording.sim_trace import VIZ_DIR_ENV, ChunkedLog, SimTrace, VizTrace  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rec(tmp_path):
    REC.__init__()
    REC.start_run("test_run", root=tmp_path)
    yield REC
    REC.__init__()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# ---------------------------------------------------------------------- recorder core


def test_commands_are_numbered_finalized_and_never_overwritten(rec, tmp_path):
    first = rec.begin_command("Form a circle!")
    rec.record_action("initial_prompt", "ok")
    second = rec.begin_command("Spin")
    assert first.name == "0001_form_a_circle" and second.name == "0002_spin"
    result = json.loads((first / "result.json").read_text())
    assert result["status"] == "completed" and result["finalized_by"] == "next_command"
    assert rec.save_json(second / "x.json", {"a": 1}).name == "x.json"
    assert rec.save_json(second / "x.json", {"a": 2}).name == "x_1.json"
    assert json.loads((second / "x.json").read_text()) == {"a": 1}


def test_open_command_is_marked_interrupted_on_restart(rec, tmp_path):
    open_dir = rec.begin_command("crash here")
    rec.event("something", value=np.float32(1.5))
    restarted = Recorder()
    restarted.start_run("test_run", root=tmp_path)
    result = json.loads((open_dir / "result.json").read_text())
    assert result["status"] == "interrupted"
    assert restarted.begin_command("next").name.startswith("0002_")
    events = read_jsonl(open_dir / "events.jsonl")
    assert [e["event"] for e in events] == ["command_start", "something"]


def test_stage_records_exception_and_reraises(rec):
    rec.begin_command("x")
    with pytest.raises(ValueError):
        with rec.stage("outer"), rec.stage("inner"):
            raise ValueError("boom")
    events = read_jsonl(rec.command_dir / "events.jsonl")
    error = next(e for e in events if e["event"] == "stage_error")
    assert error["stage"] == "outer/inner" and error["error"]["type"] == "ValueError"
    assert "Traceback" in error["error"]["traceback"]


def test_unnamed_run(tmp_path):
    r = Recorder()
    run_dir = r.start_run(None, root=tmp_path)
    assert run_dir.name.startswith("unnamed_") and (run_dir / "run.json").exists()


# ---------------------------------------------------------------------- LLM calls


def fake_openai_client(content="hello", error=None):
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    def create(**kwargs):
        if error is not None:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            model_dump=lambda mode: {"id": "r1", "model": "m", "usage": usage,
                                     "choices": [{"message": {"content": content}}]},
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_llm_calls_are_recorded_and_returned_unchanged(rec):
    rec.begin_command("x")
    client = fake_openai_client()
    install_openai_hook(client)
    install_openai_hook(client)  # idempotent
    with rec.stage("router"):
        response = client.chat.completions.create(model="m", messages=[{"role": "user",
                                                                       "content": "hi"}])
    assert response.choices[0].message.content == "hello"
    [call] = read_jsonl(rec.command_dir / "llm_calls.jsonl")
    assert call["stage"] == "router" and call["usage"]["total_tokens"] == 5
    assert call["request"]["messages"][0]["content"] == "hi" and call["latency_s"] >= 0

    failing = fake_openai_client(error=RuntimeError("api down"))
    install_openai_hook(failing)
    with pytest.raises(RuntimeError, match="api down"):
        failing.chat.completions.create(model="m", messages=[])
    assert read_jsonl(rec.command_dir / "llm_calls.jsonl")[-1]["error"]["type"] == "RuntimeError"


# ---------------------------------------------------------------------- library snapshots


def test_library_changes_detect_registered_primitive(tmp_path):
    from swarm_gpt.core.agent_b.function_node import FunctionNode
    from swarm_gpt.core.agent_b.primitive_writer import write_descriptions_to_yaml, write_to_file

    files = {}
    for name, src in library.LIBRARY_FILES.items():
        files[name] = tmp_path / name
        shutil.copy(src, files[name])
    before = library.snapshot(tmp_path / "lib", files)
    node = FunctionNode("hover_test", "Hover in place for a while.")
    node.body = "def hover_test(params, swarm_pos, tstart, tend, limits):\n    return swarm_pos, {}"
    node.definition = "# n_args: 1\ndef hover_test(params, swarm_pos, tstart, tend, limits):\n"
    node.n_args = 1
    write_to_file([node], files["motion_primitives.py"])
    write_descriptions_to_yaml([node], files["prompt_generated_primitives.yaml"])
    after = library.snapshot(tmp_path / "lib", files)
    diff = library.changes(before, after)
    assert diff["changed"] and diff["functions_added"] == ["hover_test"]
    assert diff["registry_added"] == ["hover_test"]
    assert "+def hover_test" in diff["diff"]
    assert after["prompt_generated_names"][-1] == "hover_test"
    assert len(list((tmp_path / "lib" / "objects").iterdir())) == 5  # 3 + 2 changed files


# ---------------------------------------------------------------------- simulation traces


def fake_solver(n=2, K=3, N=2, seed=0):
    rng = np.random.default_rng(seed)
    s_x, s_uw = rng.normal(size=(6 * (K + 1), 6)), rng.normal(size=(6 * (K + 1), 3 * (N + 1)))
    x0, zeta = rng.normal(size=(n, 6)), rng.normal(size=(n, 3 * (N + 1)))
    full = (x0 @ s_x.T + zeta @ s_uw.T).reshape(n, K + 1, 6)
    return SimpleNamespace(
        pos=full[..., :3], x_0=x0, zeta=zeta, u_pos=rng.normal(size=(n, K, 3)),
        u_vel=rng.normal(size=(n, K, 3)), u_acc=rng.normal(size=(n, K, 3)),
        distance_matrix=np.ones((n, n, K + 1)),
        matrices=SimpleNamespace(S_x=s_x, S_u_W_input=s_uw)), full


def fake_sim(n=2):
    states = SimpleNamespace(pos=np.zeros((1, n, 3)), vel=np.ones((1, n, 3)),
                             quat=np.tile([0, 0, 0, 1.0], (1, n, 1)), ang_vel=np.zeros((1, n, 3)))
    return SimpleNamespace(n_drones=n, freq=400, control_freq=80,
                           data=SimpleNamespace(states=states,
                                                core=SimpleNamespace(steps=np.array([[5]]))))


def test_sim_trace_records_full_horizon_and_merges_chunks(rec, tmp_path):
    settings = {"axswarm": {"pos_min": [-1, -1, 0], "pos_max": [1, 1, 1]}}
    assert SimTrace.start(fake_sim(), None, settings, {}, False, None) is None  # not requested
    rec.pending_sim_dir = tmp_path / "sim"
    solver_settings = SimpleNamespace(freq=8, K=3, N=2, collision_envelope=[0.15] * 3)
    waypoints = {"time": np.array([[0.0, 1.0]] * 2)}
    trace = SimTrace.start(fake_sim(), solver_settings, settings, waypoints, False, None)
    trace.solves.chunk_rows, trace.steps.chunk_rows = 2, 3
    solver, full = fake_solver()
    for step in range(10):
        if step % 5 == 0:
            trace.record_solve(step / 80, np.zeros((2, 6)), solver, [True, False], [7, 9], 0.01)
        trace.record_step(step, np.full((1, 2, 13), step, dtype=np.float32), fake_sim())
    assert len(list((tmp_path / "sim" / "chunks").glob("*.npz"))) == 4  # 1 solve + 3 step chunks
    trace.close()
    assert not (tmp_path / "sim" / "chunks").exists()
    solves = np.load(tmp_path / "sim" / "axswarm_solves.npz")
    np.testing.assert_allclose(solves["pred_vel"][0], full[..., 3:])
    assert solves["pred_pos"].shape == (2, 2, 4, 3) and solves["iters"].tolist() == [[7, 9]] * 2
    assert float(solves["pred_pos_reconstruction_error"].max()) < 1e-9
    steps = np.load(tmp_path / "sim" / "crazyflow_states_80hz.npz")
    np.testing.assert_allclose(steps["t_sim"], (np.arange(10) + 1) / 80)
    np.testing.assert_allclose(steps["t_cmd"], np.arange(10) / 80)
    assert steps["cmd_state"][:, 0, 0].tolist() == list(range(10))
    assert steps["quat"].shape == (10, 2, 4)
    meta = json.loads((tmp_path / "sim" / "meta.json").read_text())
    assert meta["fields"]["axswarm_solves.npz"]["pred_vel"].startswith("(S,n,K+1,3)")


def test_chunked_log_keeps_data_without_close(tmp_path):
    log = ChunkedLog(tmp_path, "x", chunk_rows=2)
    for i in range(5):
        log.append(a=i)
    # the process dies here: the completed chunks are on disk
    data = [np.load(p)["a"].tolist() for p in sorted((tmp_path / "chunks").glob("x_*.npz"))]
    assert data == [[0, 1], [2, 3]]


def test_viz_trace_saves_on_sigterm_path(tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setenv(VIZ_DIR_ENV, str(tmp_path / "viz"))
    try:
        trace = VizTrace.from_env(fake_sim(), {1: None, 3: None}, 1.0)
        assert signal.getsignal(signal.SIGTERM) is not previous
        with pytest.raises(SystemExit):
            for i in range(3):
                trace.record(i / 80, np.zeros((1, 2, 13)), fake_sim())
                if i == 2:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        trace.close()  # what the script's finally block does
    finally:
        signal.signal(signal.SIGTERM, previous)
    data = np.load(tmp_path / "viz" / "visualizer_states_80hz.npz")
    assert data["pos"].shape == (3, 2, 3)
    assert json.loads((tmp_path / "viz" / "meta.json").read_text())["drone_ids"] == [1, 3]


def test_viz_script_still_compiles():
    from swarm_gpt.core.backend import _VIZ_SCRIPT

    compile(_VIZ_SCRIPT, "viz_script", "exec")


# ---------------------------------------------------------------------- pipeline hooks


@pytest.fixture(scope="module")
def installed():
    from swarm_gpt import recording

    recording._install_hooks()


def test_wrapped_methods_keep_signatures_and_self_correct_target(installed):
    from swarm_gpt.core.backend import AppBackend

    assert list(inspect.signature(AppBackend.initial_prompt).parameters) == [
        "self", "text", "response"]
    assert list(inspect.signature(AppBackend.reprompt).parameters) == ["self", "message"]
    # self_correct calls self.reprompt.__wrapped__: it must stay the undecorated function
    assert AppBackend.reprompt.__wrapped__.__name__ == "reprompt"
    assert not getattr(AppBackend.reprompt.__wrapped__, "_swarmgpt_recorded", False)
    assert inspect.isgeneratorfunction(AppBackend.simulate)


def test_code_generator_versions_rounds_and_llm_calls(rec, installed, monkeypatch, tmp_path):
    from swarm_gpt.core.agent_b import action, describe_primitive, pipeline

    from .test_agent_b_test_params import VALID, FakeClient, scripted_inference

    client = FakeClient()
    install_openai_hook(client)
    monkeypatch.setattr(action, "client", client)
    monkeypatch.setattr(describe_primitive, "client", client)
    monkeypatch.setattr(pipeline, "_STAGING_DIR", tmp_path / "staging")
    scripted_inference(monkeypatch, {k: [v] for k, v in VALID.items()})
    rec.begin_command("form a test line")
    with rec.stage("initial_prompt"):
        [node] = pipeline.Pipeline().run(["form a test line"])
    assert node.name == "form_test_line"
    d = rec.command_dir / "codegen" / "primitive_000"
    names = sorted(p.name for p in d.iterdir())
    for expected in ("v00_design.py", "v01_write.py", "v02_code_review.py", "describe.json",
                     "name_validation.json", "test_params.json", "certification_round_00.py",
                     "certification_rounds.jsonl", "certified.py", "result.json", "input.json"):
        assert expected in names
    rounds = read_jsonl(d / "certification_rounds.jsonl")
    assert [r["check"] for r in rounds] == ["start", "pylint", "mechanical", "runtime"]
    assert all(r.get("passed", True) for r in rounds)
    result = json.loads((d / "result.json").read_text())
    assert result["status"] == "certified" and result["n_args"] == 3
    assert (d / "certified.py").read_text() == node.body
    stages = [c["stage"] for c in read_jsonl(rec.command_dir / "llm_calls.jsonl")]
    assert any(s.endswith("primitive/describe") for s in stages)
    assert any(s.endswith("primitive/design") for s in stages)
    assert any(s.endswith("primitive/code_review") for s in stages)


def test_code_generator_failure_keeps_every_version(rec, installed, monkeypatch, tmp_path):
    from swarm_gpt.core.agent_b import action, describe_primitive, pipeline

    from .test_agent_b_test_params import BODY, VALID, FakeClient, scripted_inference

    broken = BODY.replace("final_pos[drone_ids] = target", "final_pos[drone_ids] = undefined_x")
    client = FakeClient(write=broken, debug=broken)
    monkeypatch.setattr(action, "client", client)
    monkeypatch.setattr(describe_primitive, "client", client)
    monkeypatch.setattr(pipeline, "_STAGING_DIR", tmp_path / "staging")
    scripted_inference(monkeypatch, {k: [v] for k, v in VALID.items()})
    rec.begin_command("form a broken line")
    with pytest.raises(RuntimeError, match="Certification failed"):
        pipeline.Pipeline().run(["form a broken line"])
    d = rec.command_dir / "codegen" / "primitive_000"
    assert json.loads((d / "result.json").read_text())["status"] == "failed"
    assert (d / "last_code.py").exists() and not (d / "certified.py").exists()
    debug_versions = sorted(p.name for p in d.glob("v*_debug_*.py"))
    assert len(debug_versions) == pipeline._MAX_DEBUG_CALLS
    rounds = read_jsonl(d / "certification_rounds.jsonl")
    assert sum(r["check"] == "start" for r in rounds) == pipeline._MAX_DEBUG_CALLS + 1
    assert any(r["check"] == "pylint" and r["errors"] for r in rounds)


def test_router_and_choreographer_are_recorded(rec, installed, monkeypatch):
    from swarm_gpt.core.choreographer import Choreographer

    choreo = Choreographer(ROOT / "config/crazyflies.yaml")  # read only
    client = fake_openai_client(json.dumps({"parts": [
        {"command": "move drone 1 up", "primitive": "move", "reasoning": "covered"}]}))
    install_openai_hook(client)
    choreo.router.client = client
    rec.drone_ids = dict(choreo.agents)
    rec.begin_command("move drone 1 up")
    assert choreo.analyze_command("move drone 1 up") == {"routing": "existing_system",
                                                        "missing": []}
    text = "```yaml\nchoreography:\n1: move(0, 0, 120, 1)\n2: PLAN\nEND\n```"
    monkeypatch.setattr(choreo, "_call_openai", lambda messages: text)
    choreo.generate_choreography(choreo.format_initial_prompt("move drone 1 up"))
    waypoints = choreo.response2waypoints(text)
    router = json.loads((rec.command_dir / "router" / "router.json").read_text())
    assert router["output"]["routing"] == "existing_system" and "move" in router["catalog"]
    d = rec.command_dir / "choreography" / "attempt_00"
    assert (d / "response.txt").read_text() == text
    parse = json.loads((d / "parse.json").read_text())
    assert parse["status"] == "ok" and parse["sliced_steps"][0]["1"] == "move(0, 0, 120, 1)"
    [call] = parse["primitive_calls"]
    assert call["primitive"] == "move" and call["swarm_pos_out_cm"][0] == [0, 0, 120]
    saved = np.load(d / "waypoints.npz")
    np.testing.assert_array_equal(saved["pos"], waypoints["pos"])
    assert saved["drone_ids"].tolist() == [1, 3, 6, 8]
    assert (d / "waypoints_before_clip.npz").exists()
    with pytest.raises(Exception):
        choreo.response2waypoints("no choreography here")
    failed = json.loads((rec.command_dir / "choreography" / "attempt_01" / "parse.json"
                         ).read_text())
    assert failed["status"] == "error" and failed["error"]["type"] == "LLMFormatError"


# ---------------------------------------------------------------------- backend end to end


class FakeChoreographer:
    def __init__(self, fail_first=False):
        self.agents, self.starting_pos = {0: 1, 1: 3}, {0: np.zeros(3), 1: np.ones(3)}
        self.messages, self.fail_first, self.num_drones = [], fail_first, 2

    def reset_history(self):
        self.messages.clear()

    def analyze_command(self, text):
        return {"routing": "existing_system", "missing": []}

    def format_initial_prompt(self, text):
        return [{"role": "user", "content": text}]

    def format_reprompt(self, message):
        return [{"role": "user", "content": message}]

    def generate_choreography(self, prompt):
        self.messages.extend(prompt)
        self.messages.append({"role": "assistant", "content": "ok"})
        return "ok"

    def response2waypoints(self, text, strict=True):
        from swarm_gpt.exception import LLMException

        if self.fail_first:
            self.fail_first = False
            raise LLMException("too close")
        t = np.tile(np.array([0.0, 1.0]), (2, 1))
        pos = np.stack([np.array([[0.0, 0, 1], [0.5, 0, 1]]), np.array([[1.0, 0, 1], [1.0, 0.5, 1]])])
        return {"time": t, "pos": pos, "vel": np.zeros_like(pos), "acc": np.zeros_like(pos)}


class FakeTime:
    def __init__(self):
        self.t, self.nextTime = 0.0, None

    def time(self):
        return self.t

    def sleep(self, duration):
        self.t += duration

    def sleepForRate(self, rate):
        time.sleep(0.01)
        self.t += 1.0 / rate

    def isShutdown(self):
        return False


class FakeCF:
    def __init__(self, i):
        self.prefix, self.initialPosition, self.sent = f"/cf{i}", np.array([i, 0.0, 0.0]), []
        self.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=time.time_ns())))
        p, q = SimpleNamespace(x=i, y=0.0, z=1.0), SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        self.poseStamped = {"timestamp_sec": 1, "timestamp_nsec": 2,
                            "pose": SimpleNamespace(position=p, orientation=q)}
        self.status = {"battery": 4.1}

    def cmdFullState(self, pos, vel, acc, yaw, omega):
        self.sent.append((pos, vel))

    def goTo(self, *a, **k):
        pass

    def arm(self, arm=True):
        pass

    def notifySetpointsStop(self, *a, **k):
        pass


def fake_simulate_axswarm(waypoints, settings, gui=False):
    n, n_steps = 2, 80
    rec_sim = fake_sim(n)
    trace = SimTrace.start(rec_sim, SimpleNamespace(freq=8, K=3, N=2,
                                                    collision_envelope=[0.15] * 3),
                           settings, waypoints, False, None)
    solver, _ = fake_solver(n)
    controls = []
    for step in range(n_steps):
        yield "progress", step + 1, n_steps
        if step % 10 == 0:
            trace.record_solve(step / 80, np.zeros((n, 6)), solver, [True] * n, [1] * n, 0.01)
            controls.append(np.concatenate([waypoints["pos"][:, 0] + step / 800,
                                            np.zeros((n, 3))], -1))
        trace.record_step(step, np.zeros((1, n, 13)), rec_sim)
    trace.close()
    controls = np.stack(controls)
    k = len(controls)
    yield "result", {
        "num_drones": n, "log_freq": 8, "sim_freq": 400, "timestamps": np.arange(n_steps) / 80,
        "states": controls, "controls": controls, "waypoints": waypoints,
        "simulation_freq": 400, "amswarm_every_n_steps": 10, "solve_times": np.full(k, 0.01),
        "phase1": {"timestamps": np.arange(1, k) / 8, "h_min_actual": np.ones(k - 1),
                   "predicted_safe_actual_unsafe": 0, "predicted_unsafe_actual_safe": 0},
    }, "placeholder"


def make_backend(choreo):
    import yaml

    from swarm_gpt.core.backend import AppBackend
    from swarm_gpt.core.drone_controller import DroneController

    app = AppBackend.__new__(AppBackend)
    app.settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    app.waypoints, app.splines, app.choreographer = None, {}, choreo
    app._strict_processing, app._viz_proc, app._viz_data_path = True, None, None
    dc = DroneController.__new__(DroneController)
    dc.freq, dc._ros_running, dc._node, dc._last_cmd_pos, dc._drone_pos = 10, True, None, {}, {}
    cfs = {1: FakeCF(1), 3: FakeCF(3)}
    allcfs = SimpleNamespace(crazyfliesById=cfs, takeoff=lambda **k: None,
                             get_clock=cfs[1].node.get_clock)
    dc.swarm = SimpleNamespace(allcfs=allcfs, timeHelper=FakeTime())
    app.drone_controller = dc
    return app, cfs


def test_backend_command_simulation_and_deployment(rec, installed, monkeypatch):
    from swarm_gpt.core import backend as backend_module
    from swarm_gpt.core import drone_controller as dc_module
    from swarm_gpt.recording import deploy

    monkeypatch.setattr(backend_module, "simulate_axswarm", fake_simulate_axswarm)
    monkeypatch.setattr(dc_module.rclpy, "spin_once", lambda *a, **k: None)
    monkeypatch.setattr(dc_module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(deploy.shutil, "which", lambda name: None)  # no real ros2 bag in tests
    app, cfs = make_backend(FakeChoreographer(fail_first=True))

    messages = app.initial_prompt("make a line")  # LLMException -> self_correct -> reprompt
    assert messages[-1] == {"role": "assistant", "content": "ok"}
    command = rec.command_dir
    assert (command / "library_before.json").exists()

    sim_data = None
    gen = app.simulate(gui=False)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        sim_data = stop.value
    assert sim_data["num_drones"] == 2 and set(app.splines) == {1, 3}
    sim_dir = command / "simulation_000"
    for name in ("meta.json", "axswarm_solves.npz", "crazyflow_states_80hz.npz", "summary.json",
                 "waypoints.npz", "axswarm_reference_8hz.npz", "prediction_and_safety.npz",
                 "smoothed_splines.npz", "smoothed_splines.json", "input_waypoints.npz",
                 "status.json"):
        assert (sim_dir / name).exists(), name
    splines = np.load(sim_dir / "smoothed_splines.npz")
    np.testing.assert_allclose(splines["pos_80hz"][:, 0, 0],
                               app.splines[1][0](splines["t_80hz"]))
    assert splines["drone_ids"].tolist() == [1, 3]

    app.deploy()
    deploy_dir = command / "deployment_000"
    cmds = np.load(deploy_dir / "hardware_commands.npz")
    meta = json.loads((deploy_dir / "meta.json").read_text())
    phases = [meta["phases"][i] for i in np.unique(cmds["phase"])]
    assert set(phases) == {"run_spline_trajectories", "land_full_state"}
    assert len(cmds["t_wall_ns"]) == sum(len(cf.sent) for cf in cfs.values())
    first = int(np.argmax(cmds["drone_id"] == 1))
    np.testing.assert_allclose(cmds["cmd_pos"][first], cfs[1].sent[0][0])
    assert cmds["pose_pos"][first].tolist() == [1, 0, 1] and cmds["battery_v"][first] == 4.1
    traj = cmds["phase"] == meta["phases"].index("run_spline_trajectories")
    assert np.isfinite(cmds["spline_t"][traj]).all() and np.isnan(cmds["spline_t"][~traj]).all()
    for drone, t, pos in zip(cmds["drone_id"][traj], cmds["spline_t"][traj], cmds["cmd_pos"][traj]):
        np.testing.assert_allclose(pos, [s(t) for s in app.splines[int(drone)]])  # exact
    tstart = meta["spline_tstart_mono_ns"]
    assert 0 <= (cmds["t_mono_ns"][traj][0] - tstart) / 1e9 - cmds["spline_t"][traj][0] < 0.01
    assert meta["splines_source_simulation"] == str(sim_dir)
    assert meta["crazyflies"]["1"]["status_at_start"]["battery"] == 4.1
    labels = [a["label"] for a in meta["clock_anchors"]]
    assert labels[0] == "deploy_start" and labels[-1] == "deploy_end"
    assert "run_spline_trajectories_start" in labels
    assert all(a["t_ros_ns"] and a["sampling_latency_ns"] >= 0 for a in meta["clock_anchors"])
    events = read_jsonl(command / "events.jsonl")
    started = [e["phase"] for e in events if e["event"] == "flight_phase_start"]
    assert started == ["arm", "takeoff", "move_to_initial_positions", "run_spline_trajectories",
                       "land_full_state"]
    assert any(e["event"] == "rosbag_unavailable" for e in events)

    app.reset_data()  # "New Command" button
    assert rec.command_dir is None
    result = json.loads((command / "result.json").read_text())
    assert result["finalized_by"] == "new_command_button"
    assert [a["action"] for a in result["actions"]] == ["initial_prompt", "simulate", "deploy"]
    assert result["status"] == "completed" and result["library_changed"] is False
    assert (command / "library_changes.json").exists()


def test_failed_command_is_recorded(rec, installed):
    app, _ = make_backend(FakeChoreographer())
    app.choreographer.analyze_command = lambda text: (_ for _ in ()).throw(KeyError("router"))
    with pytest.raises(KeyError):
        app.initial_prompt("break")
    rec.end_command("test")
    command = sorted((rec.run_dir / "commands").iterdir())[-1]
    result = json.loads((command / "result.json").read_text())
    assert result["status"] == "failed" and result["errors"][0]["error"]["type"] == "KeyError"


LAUNCH_SCRIPT = """
import json, sys, types
from types import SimpleNamespace
sys.modules["swarm_gpt.ui"] = types.SimpleNamespace(create_ui=None)  # UI replaced below
import swarm_gpt.launch as launch
from swarm_gpt.recording import recorder
from swarm_gpt.core import _llm_client
from swarm_gpt.core.backend import AppBackend
from swarm_gpt.core.sim import simulate_axswarm

recorder.DEFAULT_RECORD_ROOT = __import__("pathlib").Path(sys.argv[1])
launch.AppBackend = lambda **kwargs: SimpleNamespace(**kwargs)
launch.create_ui = lambda backend: SimpleNamespace(launch=lambda: None)
launch.main(run=None if sys.argv[2] == "none" else sys.argv[2])
print(json.dumps({
    "recording_active": recorder.REC.active,
    "backend_wrapped": getattr(AppBackend.initial_prompt, "_swarmgpt_recorded", False),
    "llm_wrapped": getattr(_llm_client.client.chat.completions.create, "_swarmgpt_recorded", False),
    "files": sorted(p.name for p in recorder.DEFAULT_RECORD_ROOT.glob("*")),
}))
"""


@pytest.mark.parametrize("run", [None, "run_x"])
def test_launch_records_only_with_a_run_name(tmp_path, run):
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", LAUNCH_SCRIPT, str(tmp_path), run or "none"],
                         capture_output=True, text=True, timeout=300, cwd=ROOT,
                         env={**os.environ, "OPENAI_API_KEY": "test-key-not-used"})
    state = json.loads(out.stdout.strip().splitlines()[-1])
    if run is None:  # original behavior: no hooks, no files
        assert state == {"recording_active": False, "backend_wrapped": False,
                         "llm_wrapped": False, "files": []}
    else:
        assert state == {"recording_active": True, "backend_wrapped": True, "llm_wrapped": True,
                         "files": ["run_x"]}
