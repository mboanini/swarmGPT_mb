# Agent 2: esegue la pipeline GenSwarm per comandi non gestibili dalle motion primitives.

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class GenSwarmAgent:
    def __init__(self, workspace: Path, model_id: str = "gpt-4o"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._imports_done = False

        # Trova GenSwarm root (3 livelli su da swarm_gpt/core/)
        repo_root = Path(__file__).resolve().parents[2]
        self._genswarm_root = repo_root / "GenSwarm"

        if not self._genswarm_root.exists():
            raise FileNotFoundError(f"GenSwarm submodule non trovato in {self._genswarm_root}")

        # Aggiungi GenSwarm e repo root al path (genswarm_overrides/ è nella root)
        repo_root_str = str(repo_root.resolve())
        genswarm_str = str(self._genswarm_root.resolve())
        if genswarm_str not in sys.path:
            sys.path.insert(0, genswarm_str)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        # Registra il modulo reale swarmgpt_global_apis in sys.modules prima che GenSwarm avvii
        # la pipeline, così il GrammarCheck non fallisce con ImportError sul codice generato.
        # Il modulo reale restituisce defaults sicuri finché init() non viene chiamato con il
        # backend Crazyflie (_allcfs is None → no-op / valori default).
        if "swarmgpt_global_apis" not in sys.modules:
            import genswarm_overrides.swarmgpt_global_apis as _real_global
            sys.modules["swarmgpt_global_apis"] = _real_global

        # Configura llm_config.yaml usando OPENAI_API_KEY
        self._setup_llm_config()

    def _setup_llm_config(self):
        """Crea config/llm_config.yaml in GenSwarm con la API key dall'ambiente."""
        import os
        import yaml

        api_key = os.getenv("OPENAI_API_KEY", "")
        config = {
            "api_base": {"GPT": "https://api.openai.com/v1"},
            "api_key":  {"GPT": api_key},
            "model":    {"GPT": self.model_id},
        }
        config_path = self._genswarm_root / "config" / "llm_config.yaml"
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def _lazy_import(self):
        if self._imports_done:
            return

        global WorkflowContext, AnalyzeConstraints, AnalyzeSkills
        global GenerateFunctions, root_manager

        from modules.framework.context import WorkflowContext
        from modules.framework.actions import AnalyzeConstraints, AnalyzeSkills
        from modules.framework.actions import GenerateFunctions
        from modules.utils import root_manager

        self._imports_done = True

    async def run(self, command: str, task_name: str = "free") -> tuple[str, str]:
        """Esegue la pipeline: AnalyzeConstraints → AnalyzeSkills → GenerateFunctions."""
        self._lazy_import()

        logger.info(f"[Agent 2] Pipeline per: '{command}'")

        # Cancella la cache pkl dal run precedente: GenSwarm marca i nodi come
        # State.CHECKED in base ai file .pkl sul disco. Se esistono, salta il
        # rieseguimento dell'LLM e riusa constraints/skills del task precedente.
        for pkl_file in self.workspace.glob("*.pkl"):
            try:
                pkl_file.unlink()
                logger.debug("[Agent 2] Cache invalidata: %s", pkl_file.name)
            except OSError as e:
                logger.warning("[Agent 2] Impossibile eliminare %s: %s", pkl_file.name, e)

        root_manager.update_root(str(self.workspace))

        # Reset singleton tra run diversi
        WorkflowContext._instance = None
        args = argparse.Namespace(
            llm_name="GPT",
            run_experiment_name=[task_name],
            interaction_mode=False,
            print_to_terminal=True,
            generate_mode="layer",
            prompt_type="default",
        )
        context = WorkflowContext(args=args)
        context.command = command

        logger.info("[Agent 2] Step 1/3: Analyze Constraints")
        await AnalyzeConstraints("").run(auto_next=False)

        logger.info("[Agent 2] Step 2/3: Analyze Skills")
        await AnalyzeSkills("").run(auto_next=False)

        logger.info("[Agent 2] Step 3/3: Generate Functions (no W   riteRun)")
        step3 = GenerateFunctions()
        async def _run_no_writerun():
            from modules.framework.code import State
            finish = False
            while not finish:
                await asyncio.sleep(1)
                try:
                    await step3._actions.run_internal_actions()
                except SystemExit:
                    logger.warning("[Agent 2] SystemExit in run_internal_actions, stopping early")
                    break
                finish = all(node.state == State.CHECKED for node in step3.skill_tree.nodes)
            step3.skill_tree.save_functions_to_file()
            logger.info("[Agent 2] Code generated, WriteRun skipped")
        step3._run = _run_no_writerun
        try:
            await step3.run(auto_next=False)
        except SystemExit:
            logger.warning("[Agent 2] GenerateFunctions terminato anticipatamente (SystemExit)")

        global_code = self._read_file("global_skill.py")
        local_code  = self._read_file("local_skill.py")
        logger.info(f"[Agent 2] Done. global={len(global_code)}ch, local={len(local_code)}ch")
        return global_code, local_code

    def _read_file(self, filename: str) -> str:
        path = self.workspace / filename
        return path.read_text(encoding="utf-8") if path.exists() else ""