"""Read-only snapshots of the primitive library (motion_primitives.py and the two YAML files).

The recorder never writes to the library files. File contents are stored once per unique content in
experiments/<run>/library/objects/<sha256>.<ext>; manifests reference them by hash.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import logging
import re
from pathlib import Path

import yaml

from swarm_gpt.recording.recorder import ROOT, now_ns

logger = logging.getLogger(__name__)

LIBRARY_FILES = {
    "motion_primitives.py": ROOT / "swarm_gpt/core/motion_primitives.py",
    "primitives.yaml": ROOT / "swarm_gpt/data/primitives.yaml",
    "prompt_generated_primitives.yaml": ROOT / "swarm_gpt/data/prompt_generated_primitives.yaml",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_files(files: dict[str, Path] | None = None) -> dict[str, str | None]:
    """Current text of every library file (None if missing)."""
    out = {}
    for name, path in (files or LIBRARY_FILES).items():
        try:
            out[name] = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            out[name] = None
    return out


def parse_primitives_module(source: str) -> dict:
    """Entries of the motion_primitives dict and a hash of every top-level function's source."""
    tree = ast.parse(source)
    registry, functions = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "motion_primitives" for t in node.targets
        ):
            try:
                registry = ast.literal_eval(node.value)
            except ValueError:
                registry = {"<unparseable>": ast.get_source_segment(source, node.value)}
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = sha256(ast.get_source_segment(source, node) or "")
    return {"registry": registry, "functions": functions}


def manifest(texts: dict[str, str | None]) -> dict:
    """Machine-readable description of the library state."""
    out = {"files": {n: (sha256(t) if t is not None else None) for n, t in texts.items()}}
    try:
        out["module"] = parse_primitives_module(texts["motion_primitives.py"] or "")
    except SyntaxError as e:
        out["module"] = {"error": f"SyntaxError: {e}"}
    try:
        out["router_catalog"] = yaml.safe_load(texts["primitives.yaml"] or "") or {}
    except yaml.YAMLError as e:
        out["router_catalog"] = {"error": str(e)}
    try:
        content = (yaml.safe_load(texts["prompt_generated_primitives.yaml"] or "") or {}).get(
            "content"
        ) or ""
        out["prompt_generated_names"] = re.findall(r"^  (\w+)\(", content, re.MULTILINE)
    except (yaml.YAMLError, AttributeError) as e:
        out["prompt_generated_names"] = {"error": str(e)}
    out["library_hash"] = sha256("".join(f"{n}:{h}" for n, h in sorted(out["files"].items())))
    return out


def snapshot(library_dir: Path, files: dict[str, Path] | None = None) -> dict:
    """Store the current file contents (content-addressed) and return the manifest."""
    texts = read_files(files)
    objects = Path(library_dir) / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        if text is None:
            continue
        path = objects / f"{sha256(text)}{Path(name).suffix}"
        if not path.exists():
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
    return {**manifest(texts), "texts": texts, **now_ns()}


def changes(before: dict, after: dict) -> dict:
    """Primitives added/removed/modified between two snapshots, plus a unified diff of the files."""
    out = {"library_hash_before": before["library_hash"],
           "library_hash_after": after["library_hash"],
           "changed": before["library_hash"] != after["library_hash"]}
    fb = before.get("module", {}).get("functions", {})
    fa = after.get("module", {}).get("functions", {})
    out["functions_added"] = sorted(set(fa) - set(fb))
    out["functions_removed"] = sorted(set(fb) - set(fa))
    out["functions_modified"] = sorted(n for n in set(fa) & set(fb) if fa[n] != fb[n])
    rb = before.get("module", {}).get("registry", {})
    ra = after.get("module", {}).get("registry", {})
    out["registry_added"] = sorted(set(ra) - set(rb))
    out["registry_modified"] = sorted(n for n in set(ra) & set(rb) if ra[n] != rb[n])
    cb, ca = before.get("router_catalog", {}), after.get("router_catalog", {})
    if isinstance(cb, dict) and isinstance(ca, dict):
        out["router_catalog_added"] = sorted(set(ca) - set(cb))
        out["router_catalog_modified"] = sorted(n for n in set(ca) & set(cb) if ca[n] != cb[n])
    diff = []
    for name in LIBRARY_FILES:
        a = (before["texts"].get(name) or "").splitlines(keepends=True)
        b = (after["texts"].get(name) or "").splitlines(keepends=True)
        diff.extend(difflib.unified_diff(a, b, f"before/{name}", f"after/{name}"))
    out["diff"] = "".join(diff)
    return out


def without_texts(snap: dict) -> dict:
    """Manifest without the inlined file texts (those live in objects/)."""
    return {k: v for k, v in snap.items() if k != "texts"}
