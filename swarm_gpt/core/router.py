"""
Router for drone swarm commands.

Architecture:
Direct LLM classification (gpt-4o-mini) — given the full primitive catalog
(name + description) and the command in one prompt, the LLM decides whether
each part of the command (it may describe several distinct motions/formations)
is covered by an existing primitive, and returns the exact wording of every
part that is NOT covered.

Returns:
    {
        "routing": "existing_system" | "code_generation",
        "missing": []  # list of sub-commands not covered by existing primitives
    }
"""

import json
import logging
from pathlib import Path

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)


class Router:
    def __init__(self, primitives_path: str, openai_client: OpenAI = None):
        """
        Args:
            primitives_path: Path to primitives.yaml
            openai_client:   OpenAI client (creates one if not provided)
        """
        self.client = openai_client or OpenAI()
        self._primitives_path = primitives_path

        # Load primitives
        with open(primitives_path) as f:
            self.primitives: dict = yaml.safe_load(f)

        # Load prompt
        _router_prompt_path = Path(__file__).resolve().parents[1] / "data/router_prompt.yaml"
        with open(_router_prompt_path) as f:
            self.prompt_router = yaml.safe_load(f)

        logger.info("Router ready with %d primitives", len(self.primitives))

    def route(self, command: str) -> dict:
        """
        Route a command.

        Returns:
            {
                "routing": "existing_system" | "code_generation",
                "missing": ["sub-command 1", ...]   # empty if existing_system
            }
        """
        missing = self._classify(command)

        routing = "code_generation" if missing else "existing_system"
        result = {"routing": routing, "missing": missing}
        print(f"[router] '{command}' -> routing={routing}, missing={missing}")
        logger.info("Routing result: %s", result)
        return result

    def register_primitive(
        self,
        name: str,
        description: str,
        n_args: int,
    ) -> None:
        """
        Register a new primitive at runtime.
        Called by the code generator after creating a new primitive.

        Args:
            name:        Primitive name
            description: Short description
            n_args:      Number of arguments the primitive accepts
        """
        self.primitives[name] = {
            "description": description,
            "n_args": n_args,
        }

        with open(self._primitives_path, "w") as f:
            yaml.safe_dump(self.primitives, f, allow_unicode=True, sort_keys=False)

        logger.info("Registered new primitive '%s'", name)

    def _build_catalog(self) -> str:
        return "\n".join(
            f"- {name}: {data['description']}" for name, data in self.primitives.items()
        )

    def _classify(self, command: str) -> list[str]:
        """
        Asks the LLM to break the command into its distinct action(s) and,
        for each one, decide which primitive covers it (or null if none do).
        Returns the exact wording of each uncovered part (empty list if the
        command is fully covered).
        """
        user_content = self.prompt_router["user_initial"].format(
            catalog=self._build_catalog(),
            command=command,
        )
        messages = [
            {"role": "system", "content": self.prompt_router["system_initial"]},
            {"role": "user", "content": user_content},
            {"role": "system", "content": self.prompt_router["example"]},
            {"role": "system", "content": self.prompt_router["output_format"]},
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        parts = data.get("parts", [])
        for part in parts:
            primitive = part.get("primitive")
            status = primitive if primitive else "NOT COVERED"
            print(f"  [router]   part: '{part.get('command')}' -> {status}  ({part.get('reasoning')})")
        missing = [part["command"] for part in parts if part.get("primitive") is None]
        logger.info("'%s' → parts: %s → missing: %s", command, parts, missing)
        return missing