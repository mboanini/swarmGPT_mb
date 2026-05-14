"""Persistent registry of dynamically generated motion primitives.

Generated functions are stored as individual .py files under generated_library/
and registered in a manifest.json. On load they are injected into the
swarm_gpt.core.motion_primitives module so that the Choreographer can call them
transparently via primitive_by_name().
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LIBRARY_DIR = Path(__file__).parent / "generated_library"
MANIFEST_FILE = LIBRARY_DIR / "manifest.json"


class DynamicLibrary:
    """Manages persistence and hot-loading of LLM-generated motion primitives."""

    def __init__(self) -> None:
        LIBRARY_DIR.mkdir(exist_ok=True)
        self._manifest: dict[str, dict] = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        fn_name: str,
        n_args: int,
        description: str,
        source_code: str,
    ) -> None:
        """Persist a new generated function and hot-load it into the runtime.

        Args:
            fn_name: Python identifier used as both file name and call name.
            n_args: Number of positional parameters the function accepts (after
                self-contained params tuple — i.e. the length of the params tuple).
            description: Human-readable description stored in the manifest.
            source_code: Complete Python source of the function (def fn_name(...):).

        Raises:
            ValueError: If fn_name already exists in the registry.
        """
        if fn_name in self._manifest:
            raise ValueError(
                f"Function '{fn_name}' is already registered. "
                "Use update() to replace it."
            )
        self._write_file(fn_name, source_code)
        self._manifest[fn_name] = {
            "n_args": n_args,
            "description": description,
            "file": f"{fn_name}.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_manifest()
        self._inject_into_module(fn_name)
        logger.info("Registered new primitive '%s' (n_args=%d)", fn_name, n_args)

    def update(
        self,
        fn_name: str,
        n_args: int,
        description: str,
        source_code: str,
    ) -> None:
        """Replace an existing generated function."""
        if fn_name not in self._manifest:
            raise KeyError(f"Function '{fn_name}' is not in the registry.")
        self._write_file(fn_name, source_code)
        self._manifest[fn_name].update(
            {
                "n_args": n_args,
                "description": description,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_manifest()
        self._inject_into_module(fn_name)
        logger.info("Updated primitive '%s'", fn_name)

    def load_all_into_module(self) -> int:
        """Import every registered function into swarm_gpt.core.motion_primitives.

        Returns:
            Number of successfully loaded functions.
        """
        loaded = 0
        for fn_name in list(self._manifest):
            try:
                self._inject_into_module(fn_name)
                loaded += 1
            except Exception as exc:
                logger.warning("Could not load '%s': %s", fn_name, exc)
        logger.info("Loaded %d generated primitive(s) into motion_primitives", loaded)
        return loaded

    def list_functions(self) -> dict[str, dict]:
        """Return a copy of the manifest."""
        return dict(self._manifest)

    def exists(self, fn_name: str) -> bool:
        return fn_name in self._manifest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_file(self, fn_name: str, source_code: str) -> None:
        path = LIBRARY_DIR / f"{fn_name}.py"
        path.write_text(source_code, encoding="utf-8")

    def _inject_into_module(self, fn_name: str) -> None:
        """Load the .py file and inject the function into the live module."""
        import swarm_gpt.core.motion_primitives as mp_module

        file_path = LIBRARY_DIR / f"{fn_name}.py"
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(
            f"_genlib_{fn_name}", file_path
        )
        mod = importlib.util.module_from_spec(spec)
        # Expose numpy and scipy as they are used by all primitives
        mod.__dict__.update(
            {
                "np": sys.modules.get("numpy") or __import__("numpy"),
                "NDArray": __import__(
                    "numpy.typing", fromlist=["NDArray"]
                ).NDArray,
            }
        )
        spec.loader.exec_module(mod)

        fn = getattr(mod, fn_name, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"Module {file_path} does not define a callable named '{fn_name}'"
            )

        # Inject into the live motion_primitives module
        setattr(mp_module, fn_name, fn)
        mp_module.motion_primitives[fn_name] = {
            "n_args": self._manifest[fn_name]["n_args"]
        }

    def _load_manifest(self) -> dict[str, dict]:
        if MANIFEST_FILE.exists():
            try:
                return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupt manifest — starting fresh")
        return {}

    def _save_manifest(self) -> None:
        MANIFEST_FILE.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
