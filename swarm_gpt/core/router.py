"""
Router for drone swarm commands.

Architecture:
1. CHUNKING (gpt-4o-mini): splits compound commands into atomic sub-commands
2. SEMANTIC CHECK (embedding similarity): checks if each chunk is covered by existing primitives
   - score >= MATCH_THRESHOLD → covered (existing_system)
   - score <  MATCH_THRESHOLD → not covered (code_generation)
   No ambiguous zone, no LLM in the routing step.

Returns:
    {
        "routing": "existing_system" | "code_generation",
        "missing": []  # list of sub-commands not covered by existing primitives
    }
"""

import json
import logging
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import yaml

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.2 # 0.55  # tune this based on your commands


class Router:
    def __init__(self, primitives_path: str, openai_client: OpenAI = None):
        """
        Args:
            primitives_path: Path to primitives.yaml
            openai_client:   OpenAI client (creates one if not provided)
        """
        self.client = openai_client or OpenAI()

        # Load primitives
        with open(primitives_path) as f:
            self.primitives: dict = yaml.safe_load(f)

        # Load primitives
        # with open("primitives.yaml") as f:
        #     primitives = yaml.safe_load(f)

        # Load prompt for chunk
        _router_prompt_path = Path(__file__).resolve().parents[1] / "data/router_prompt.yaml"
        with open(_router_prompt_path) as f:
            self.prompt_router = yaml.safe_load(f)

        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-compute embeddings once at startup
        self.prim_embeddings = {
            name: self.embedder.encode(data["description"], convert_to_tensor=True)
            for name, data in self.primitives.items()
        }
        logger.info("Semantic router ready with %d primitives", len(self.primitives))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def route(self, command: str) -> dict:
        """
        Route a (possibly compound) command.

        Returns:
            {
                "routing": "existing_system" | "code_generation",
                "missing": ["sub-command 1", ...]   # empty if existing_system
            }

        Examples:
            "make a circle"
            → {"routing": "existing_system", "missing": []}

            "make a circle and then rotate"
            → {"routing": "existing_system", "missing": []}

            "make a circle and then an isosceles triangle"
            → {"routing": "code_generation", "missing": ["an isosceles triangle"]}

            "explore the environment"
            → {"routing": "code_generation", "missing": ["explore the environment"]}
        """
        chunks = self._chunk_command(command)
        logger.info("Chunks: %s", chunks)

        missing = [chunk for chunk in chunks if not self._is_covered(chunk)]

        routing = "code_generation" if missing else "existing_system"
        result = {"routing": routing, "missing": missing}
        logger.info("Routing result: %s", result)
        return result

    # ------------------------------------------------------------------
    # STEP 1 — CHUNKING (one LLM call, gpt-4o-mini)
    # ------------------------------------------------------------------

    def _chunk_command(self, command: str) -> list[str]:
        """Split a compound command into atomic sub-commands."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_router["system"],
                },
                {"role": "user", "content": command},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("chunks", [command])

    # ------------------------------------------------------------------
    # STEP 2 — SEMANTIC CHECK (deterministic, no LLM)
    # ------------------------------------------------------------------

    def _is_covered(self, chunk: str) -> bool:
        """
        Returns True if the chunk is semantically covered by an existing primitive.
        Pure embedding similarity — no LLM involved.
        """
        chunk_emb = self.embedder.encode(chunk, convert_to_tensor=True)

        scores = {
            name: util.cos_sim(chunk_emb, emb).item()
            for name, emb in self.prim_embeddings.items()
        }

        best_name, best_score = max(scores.items(), key=lambda x: x[1])
        covered = best_score >= MATCH_THRESHOLD

        logger.info(
            "'%s' → best match: '%s' (%.3f) → %s",
            chunk, best_name, best_score,
            "COVERED" if covered else "NOT COVERED",
        )
        return covered