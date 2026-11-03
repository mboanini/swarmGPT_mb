"""Automatic experiment recording for SwarmGPT (observability only, no change of behavior).

launch.py calls install() once, before the UI is created, only when a run name is given
(`--run <name>`). Every natural-language command then gets its own record under
experiments/<run>/commands/; see recorder.py and README.md for the layout. Without install(), all
recording code stays inactive (no hooks, no files).
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

from swarm_gpt.recording import library
from swarm_gpt.recording.recorder import REC, ROOT, now_ns

logger = logging.getLogger(__name__)

CONFIG_FILES = [
    "swarm_gpt/data/settings.yaml",
    "config/crazyflies.yaml",
    "config/motion_capture.yaml",
    "swarm_gpt/data/router_prompt.yaml",
    "swarm_gpt/data/prompts_no_music.yaml",
    "swarm_gpt/data/prompts_prim.yaml",
    "pixi.toml",
    "pixi.lock",
]
REPOSITORIES = [".", "submodules/axswarm", "submodules/crazyswarm2"]
PACKAGES = ["numpy", "scipy", "jax", "jaxlib", "crazyflow", "axswarm", "mujoco", "openai",
            "gradio", "tenacity", "ollama", "einops", "pylint"]


def install(run_name: str | None = None, launch_args: dict | None = None,
            root: Path | None = None) -> Path | None:
    """Start (or resume) the run and install all hooks. Never raises."""
    try:
        run_dir = REC.start_run(run_name, root)
    except Exception:
        logger.exception("Recording disabled: could not create the run directory")
        return None
    try:
        _install_hooks()
    except Exception:
        logger.exception("Recording: could not install all hooks")
    try:
        _record_session(launch_args or {})
    except Exception:
        logger.exception("Recording: could not record the session configuration")
    atexit.register(_on_exit)
    logger.warning("Recording experiment data to %s", run_dir)
    print(f"[recording] experiment data -> {run_dir}")
    return run_dir


def _install_hooks() -> None:
    from swarm_gpt.core import _llm_client
    from swarm_gpt.core.agent_b import pipeline
    from swarm_gpt.core.agent_b.code_review import CodeReview
    from swarm_gpt.core.agent_b.debug_function import DebugFunction
    from swarm_gpt.core.agent_b.describe_primitive import DescribePrimitive
    from swarm_gpt.core.agent_b.design_function import DesignFunction
    from swarm_gpt.core.agent_b.validate_description import ValidateNameDesc
    from swarm_gpt.core.agent_b.write_function import WriteFunction
    from swarm_gpt.core.backend import AppBackend
    from swarm_gpt.core.choreographer import Choreographer
    from swarm_gpt.core.drone_controller import DroneController
    from swarm_gpt.core.router import Router
    from swarm_gpt.recording import hooks
    from swarm_gpt.recording.llm import install_openai_hook

    install_openai_hook(_llm_client.client)
    hooks.install_backend_hooks(AppBackend)
    hooks.install_drone_controller_hooks(DroneController)
    hooks.install_router_hooks(Router)
    hooks.install_choreographer_hooks(Choreographer)
    hooks.install_codegen_hooks(
        pipeline,
        {"design": DesignFunction, "write": WriteFunction, "code_review": CodeReview,
         "debug": DebugFunction},
        DescribePrimitive,
        ValidateNameDesc,
    )


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              timeout=20).stdout
    except Exception as e:
        return f"<git failed: {e}>"


def _record_session(launch_args: dict) -> None:
    session_dir = REC.run_dir / "sessions" / REC.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    repos = {}
    for rel in REPOSITORIES:
        repo = ROOT / rel
        if not (repo / ".git").exists():  # e.g. crazyswarm2 is tracked inside the main repo
            continue
        diff = _git(repo, "diff", "HEAD")
        name = rel.replace("/", "_").strip(".") or "swarmGPT"
        REC.save_text(session_dir / f"git_diff_{name}.patch", diff)
        repos[rel] = {
            "commit": _git(repo, "rev-parse", "HEAD").strip(),
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip(),
            "status_porcelain": _git(repo, "status", "--porcelain").splitlines(),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        }
    configs = {}
    for rel in CONFIG_FILES:
        path = ROOT / rel
        if path.exists():
            data = path.read_bytes()
            configs[rel] = hashlib.sha256(data).hexdigest()
            if rel != "pixi.lock":
                target = session_dir / "config" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    REC.save_json(session_dir / "session.json", {
        "session_id": REC.session_id,
        "launch_args": launch_args,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
        "packages": versions,
        "repositories": repos,
        "config_sha256": configs,
        "env": {k: os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE", "ROS_DISTRO",
                                               "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION")},
        "openai_api_key_set": "OPENAI_API_KEY" in os.environ,
        **now_ns(),
    })
    snap = library.snapshot(REC.run_dir / "library")
    initial = REC.run_dir / "library" / "initial.json"
    if not initial.exists():
        REC.save_json(initial, library.without_texts(snap))
    REC.save_json(REC.run_dir / "library" / f"session_{REC.session_id}_start.json",
                  library.without_texts(snap))


def _on_exit() -> None:
    try:
        REC.end_command("application_exit")
        snap = library.snapshot(REC.run_dir / "library")
        REC.save_json(REC.run_dir / "library" / f"session_{REC.session_id}_end.json",
                      library.without_texts(snap))
        REC.event("session_end")
    except Exception:
        logger.exception("Recording: could not finalize the session")
