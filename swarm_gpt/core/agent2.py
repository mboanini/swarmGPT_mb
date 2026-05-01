"""
Agent 2 - Code Generator for SwarmGPT
=====================================
Usa i tre prompt template di GenSwarm (adattati per droni 3D).
Da salvare in: swarm_gpt/core/agent2.py

Come integrarlo in choreographer.py:
  from swarm_gpt.core.agent2 import Agent2
  self.agent2 = Agent2(call_llm_fn=self._call_openai, prompts_path=...)
  
  # Nel branch "code_generation" di backend.py:
  constraints = agent_one.get("constraints", [])
  code = self.choreographer.agent2.generate(text, constraints)
  waypoints = self.choreographer.agent2.code_to_waypoints(
      code, self.choreographer.starting_pos, n_steps=20
  )
"""

import json
import re
import logging
from pathlib import Path
import yaml
import numpy as np

logger = logging.getLogger(__name__)


# Parser (GenSwarm modules/framework/parser.py)

def parse_code(response: str) -> str:
    """Estrae codice Python da risposta LLM con markdown blocks."""
    match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def parse_json(response: str) -> dict:
    """Estrae JSON da risposta LLM."""
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    return json.loads(response.strip())


# Agent 2

class Agent2:
    """
    Genera codice di comportamento per droni Crazyflie partendo dai constraint
    estratti da Agent 1.

    Pipeline (fedele a GenSwarm):
      step1_design_functions()  →  AnalyzeSkills
      step2_write_functions()   →  WriteFunction (una call per funzione)
      step3_write_run()         →  WriteRun
      assemble()                →  file Python finale
    """

    def __init__(self, call_llm_fn, prompts_path: str = None):
        """
        Args:
            call_llm_fn: callable(messages: list[dict]) -> str
                         Il tuo _call_openai esistente.
            prompts_path: path al file prompt_agent2.yaml.
                          Default: stessa cartella di questo file.
        """
        self.call_llm = call_llm_fn

        if prompts_path is None:
            prompts_path = Path(__file__).parent.parent / "data" / "prompt_agent2.yaml"
        with open(prompts_path) as f:
            self.prompts = yaml.safe_load(f)

        self.env_des = self.prompts["env_description"]
        self.local_api = self.prompts["local_api"]
        self.global_api = self.prompts["global_api"]

    def _call(self, prompt: str) -> str:
        """Chiama LLM con un singolo messaggio utente."""
        return self.call_llm([{"role": "user", "content": prompt}])

    # Step 1: AnalyzeSkills

    def step1_design_functions(
        self, command: str, constraints: list[dict]
    ) -> list[dict]:
        """
        Da GenSwarm: AnalyzeSkills
        Progetta le firme delle funzioni a partire dai constraint.
        Restituisce lista di function specs.
        """
        constraints_str = "\n".join(
            f"- **{c['name']}**: {c['description']}"
            for c in constraints
        )

        prompt = self.prompts["analyze_skills"].format(
            env_des=self.env_des,
            instruction=command,
            constraints=constraints_str,
            local_api=self.local_api,
            global_api=self.global_api,
        )

        logger.info("[Agent2] Step 1: Designing function architecture...")
        response = self._call(prompt)

        try:
            data = parse_json(response)
            functions = data["functions"]
            names = [f["name"] for f in functions]
            logger.info(f"[Agent2] Designed {len(functions)} functions: {names}")
            return functions
        except Exception as e:
            logger.error(f"[Agent2] Step 1 parse error: {e}")
            logger.error(f"Response: {response[:400]}")
            raise

    # ── Step 2: WriteFunction ─────────────────────────────────────────────────

    def step2_write_functions(
        self,
        command: str,
        functions: list[dict],
        constraints: list[dict],
    ) -> dict[str, str]:
        """
        Da GenSwarm: WriteFunction
        Implementa ogni funzione con una call LLM separata.
        Restituisce dict {nome_funzione: codice}.
        """
        constraints_by_name = {c["name"]: c["description"] for c in constraints}
        written = {}  # costruito incrementalmente

        # Ordine: global prima, poi local (eccetto run_loop), run_loop per ultimo
        global_fns = [f for f in functions if f["scope"] == "global"]
        local_fns = [
            f for f in functions
            if f["scope"] == "local" and f["name"] != "run_loop"
        ]
        run_loop_fn = next((f for f in functions if f["name"] == "run_loop"), None)

        ordered = global_fns + local_fns
        if run_loop_fn:
            ordered.append(run_loop_fn)

        for fn in ordered:
            logger.info(f"[Agent2] Step 2: Writing {fn['name']}() [{fn['scope']}]...")

            # Constraint rilevanti per questa funzione
            fn_constraints = "\n".join(
                f"- **{name}**: {constraints_by_name.get(name, '')}"
                for name in fn.get("constraints", [])
            ) or "No specific constraints."

            # Contesto: funzioni già scritte
            other_code = "\n\n".join(written.values()) if written else "# (none yet)"

            # API per scope
            api = self.global_api if fn["scope"] == "global" else self.local_api

            # Firma semplificata (l'LLM la arricchisce in base alla description)
            sig = f"def {fn['name']}():"

            prompt = self.prompts["write_function"].format(
                env_des=self.env_des,
                instruction=command,
                robot_api=api,
                other_functions=other_code,
                constraints=fn_constraints,
                function_signature=sig,
                function_description=fn["description"],
            )

            response = self._call(prompt)
            code = parse_code(response)
            written[fn["name"]] = code

        return written

    # ── Step 3: WriteRun ──────────────────────────────────────────────────────

    def step3_write_run(
        self, command: str, written: dict[str, str]
    ) -> str:
        """
        Da GenSwarm: WriteRun
        Scrive run_loop() che orchestra tutte le funzioni.
        """
        logger.info("[Agent2] Step 3: Writing run_loop()...")

        # Se run_loop è già stato scritto in step2, usalo
        if "run_loop" in written:
            return written["run_loop"]

        all_code = "\n\n".join(written.values())

        prompt = self.prompts["write_run"].format(
            env_des=self.env_des,
            instruction=command,
            local_api=self.local_api,
            all_functions=all_code,
        )

        response = self._call(prompt)
        return parse_code(response)

    # ── Assemble ──────────────────────────────────────────────────────────────

    def assemble(self, command: str, written: dict[str, str]) -> str:
        """Assembla tutte le funzioni in un unico file Python."""
        header = f'''"""
Auto-generated drone swarm behavior
Command: {command}
Generated by Agent 2 (GenSwarm-inspired pipeline, adapted for SwarmGPT)

APIs available at runtime (injected by execution context):
  get_self_id, get_self_position, get_self_velocity,
  get_neighbors, get_all_drones_positions, get_swarm_size,
  get_environment_bounds, get_assigned_task,
  set_target_position, assign_task
"""
import numpy as np

'''
        body = "\n\n".join(written.values())
        return header + body

    # ── Pipeline principale ───────────────────────────────────────────────────

    def generate(self, command: str, constraints: list[dict]) -> str:
        """
        Pipeline completa: constraint → design → implement → assemble.

        Args:
            command: comando utente originale
            constraints: lista di {name, description} da Agent 1

        Returns:
            Codice Python completo con run_loop() e helpers.
        """
        # Step 1: design
        functions = self.step1_design_functions(command, constraints)

        # Step 2: implement
        written = self.step2_write_functions(command, functions, constraints)

        # Step 3: run_loop (se non già scritto)
        run_loop = self.step3_write_run(command, written)
        written["run_loop"] = run_loop

        # Assemble
        code = self.assemble(command, written)

        logger.info(
            f"[Agent2] Generation complete. "
            f"Functions: {list(written.keys())}"
        )
        return code

    # ── Esecuzione per waypoints ──────────────────────────────────────────────

    def code_to_waypoints(
        self,
        code: str,
        starting_pos: dict,
        n_steps: int = 20,
    ) -> np.ndarray:
        """
        Esegue il codice generato e produce waypoints per AMSwarm.

        Args:
            code: codice Python generato con run_loop()
            starting_pos: {drone_id: np.ndarray([x,y,z])} da Choreographer
            n_steps: numero di waypoints da generare

        Returns:
            np.ndarray shape (n_drones, n_steps, 3) in meters
        """
        n_drones = len(starting_pos)
        all_positions = {i: pos.copy() for i, pos in starting_pos.items()}
        assigned_tasks = {}
        waypoints_per_drone = {i: [] for i in range(n_drones)}

        def make_api(drone_id: int):
            """Costruisce il dizionario API per un singolo drone."""

            def get_self_id(): return drone_id
            def get_self_position(): return all_positions[drone_id].copy()
            def get_self_velocity(): return np.zeros(3)
            def get_swarm_size(): return n_drones
            def get_all_drones_positions():
                return {i: p.copy() for i, p in all_positions.items()}
            def get_neighbors(radius_m=2.0):
                pos = all_positions[drone_id]
                return [
                    {
                        "id": i,
                        "position": p.copy(),
                        "velocity": np.zeros(3),
                        "distance": float(np.linalg.norm(p - pos))
                    }
                    for i, p in all_positions.items()
                    if i != drone_id and np.linalg.norm(p - pos) <= radius_m
                ]
            def get_environment_bounds():
                return {
                    "x_min": -2.0, "x_max": 2.0,
                    "y_min": -2.0, "y_max": 2.0,
                    "z_min": 0.3,  "z_max": 2.0,
                }
            def get_assigned_task():
                return assigned_tasks.get(drone_id)
            def assign_task(did: int, task):
                assigned_tasks[did] = task

            target_store = {"pos": all_positions[drone_id].copy()}

            def set_target_position(position):
                pos = np.clip(
                    np.array(position, dtype=float),
                    [-1.8, -1.8, 0.4],
                    [1.8, 1.8, 1.8]
                )
                target_store["pos"] = pos

            def _get_target():
                return target_store["pos"]

            return {
                "get_self_id": get_self_id,
                "get_self_position": get_self_position,
                "get_self_velocity": get_self_velocity,
                "get_swarm_size": get_swarm_size,
                "get_all_drones_positions": get_all_drones_positions,
                "get_neighbors": get_neighbors,
                "get_environment_bounds": get_environment_bounds,
                "get_assigned_task": get_assigned_task,
                "assign_task": assign_task,
                "set_target_position": set_target_position,
                "_get_target": _get_target,
                "np": np,
            }

        # Esegui funzioni globali (allocatore) una volta
        global_api = make_api(0)
        global_ctx = {"np": np, **global_api}
        try:
            exec(code, global_ctx)
            # Cerca e chiama funzioni di allocazione
            for name, obj in list(global_ctx.items()):
                if (
                    callable(obj)
                    and not name.startswith("_")
                    and name not in global_api
                    and any(kw in name for kw in ["allocat", "assign", "global", "init"])
                ):
                    try:
                        obj()
                        logger.info(f"[Agent2] Called global function: {name}()")
                    except Exception as e:
                        logger.debug(f"[Agent2] Global fn {name} skipped: {e}")
        except Exception as e:
            logger.warning(f"[Agent2] Global exec warning: {e}")

        # Loop di simulazione: chiama run_loop() su ogni drone per n_steps
        for step in range(n_steps):
            new_positions = {}
            for drone_id in range(n_drones):
                api = make_api(drone_id)
                # Sincronizza task assegnati
                api["assign_task"].__code__ = (
                    lambda did, task: assigned_tasks.update({did: task})
                ).__code__
                for did, task in assigned_tasks.items():
                    api["assign_task"](did, task)

                local_ctx = {"np": np, **api}
                try:
                    exec(code, local_ctx)
                    if "run_loop" in local_ctx:
                        local_ctx["run_loop"]()
                    target = api["_get_target"]()
                except Exception as e:
                    logger.warning(
                        f"[Agent2] Drone {drone_id} step {step}: {e}"
                    )
                    target = all_positions[drone_id]

                new_positions[drone_id] = target.copy()
                waypoints_per_drone[drone_id].append(target.copy())

            all_positions = new_positions

        # Shape: (n_drones, n_steps, 3)
        return np.array([waypoints_per_drone[i] for i in range(n_drones)])