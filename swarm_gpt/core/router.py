"""
Router for drone swarm commands.

Architecture:
1. PRE-FILTER (regex): detects compound commands with explicit sequence connectives (-> deterministic split - no LLM call)
1. CHUNKING (gpt-4o-mini): fallback for implicit multi-task formulations
        called only when regex does not detect explicit connectives
2. SEMANTIC CHECK (SemanticRouter): checks if each chunk is covered by existing primitives
        -> pure embedding similarity, no LLM involved
        - route matched -> covered (existing_system)
        - route None -> not covered (code_generation)

Returns:
    {
        "routing": "existing_system" | "code_generation",
        "missing": []  # list of sub-commands not covered by existing primitives
    }
"""

import re
import json
import logging
from pathlib import Path

import yaml
from openai import OpenAI
# from sentence_transformers import SentenceTransformer, util
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder



logger = logging.getLogger(__name__)

# MATCH_THRESHOLD = 0.2 # 0.55  # tune this based on your commands

# Regex patterns for compound command detection
_SEQUENCE_CONNECTIVES = re.compile(
    r'\b('
    r'and\s+then|then|after\s+that|followed\s+by|'
    r'afterwards|subsequently|next|finally|lastly|'
    r'first.*then|before\s+that'
    r')\b',
    re.IGNORECASE
)

_LIST_SEPARATORS = re.compile(
    r',\s*(?:and\s+)?(?:then\s+)?(?=[a-z])',
    re.IGNORECASE
)

_SPLIT_PATTERN = re.compile(
    r'\s*(?:'
    r'and\s+then|then|after\s+that|followed\s+by|'
    r'afterwards|subsequently|'
    r',\s*(?:and\s+)?'
    r')\s*',
    re.IGNORECASE
)


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

        logger.info("Building semantic routes...")
        routes = []
        for name, data in self.primitives.items():
            utterances = [data["description"]] + data.get("utterances", [])
            routes.append(Route(
                name=name, 
                utterances=utterances,
                score_threshold=0.5
            ))

        encoder = OpenAIEncoder()
        self.semantic_router = SemanticRouter(
            encoder=encoder,
            routes=routes,
            auto_sync="local",
        )
        logger.info(
            "Semantic router ready with %d routes", len(routes)
        )

    def route(self, command: str) -> dict:
        """
        Route a (possibly compound) command.

        Returns:
            {
                "routing": "existing_system" | "code_generation",
                "missing": ["sub-command 1", ...]   # empty if existing_system
            }
        """
        chunks = self._chunk_command(command)
        logger.info("Chunks: %s", chunks)

        missing = [chunk for chunk in chunks if not self._is_covered(chunk)]

        routing = "code_generation" if missing else "existing_system"
        result = {"routing": routing, "missing": missing}
        logger.info("Routing result: %s", result)
        return result

    def register_primitive(
        self,
        name: str,
        description: str,
        utterances: list[str],
    ) -> None:
        """
        Register a new primitive at runtime.
        Called by the code generator after creating a new primitive
        The route is added incrementally - no restart required

        Args: 
            name:        Primitive name (e.g. "isosceles_triangle")
            description: Short description used as base utterance
            utterances:  Auto-generated utterance variants
        """
        new_route = Route(
            name=name,
            utterances=[description] + utterances,
        )
        self.semantic_router.add(new_route)

        self.primitives[name] = {
            "description": description,
            "utterances": utterances,
        }
        logger.info("Registered new primitive '%s'", name)

    # STEP 1 — CHUNKING

    def _is_compound(self, command:str) -> bool:
        """
        Detects multi-task commands via explicit sequence connectives.
        """
        if _SEQUENCE_CONNECTIVES.search(command):
            return True
        if len(_LIST_SEPARATORS.findall(command)) >= 1:
            return True
        return False


    def _chunk_command(self, command: str) -> list[str]:
        """
        Hybrid split strategy:
        - Explicit connectives detected -> regex split (no LLM call)
        - No explicit connectives -> LLM chunker
          e.g. "transitions into", "envolves into", "morphs from X to Y"

        The LLM chunker always returns at least one chunk:
        - single-intent query -> [command] (no decomposition)
        - multi-task query -> [chunk1, chunk2, ...]
        """
        if self._is_compound(command):
            chunks = _SPLIT_PATTERN.split(command)
            chunks = [c.strip() for c in chunks if c.strip()]
            logger.info("Deterministic split -> %s", chunks)
            return chunks

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
        chunks = data.get("chunks", [command])
        logger.info("LLM chunker split → %s", chunks)
        return data.get("chunks", [command])

    # STEP 2 — SEMANTIC CHECK

    def _is_covered(self, chunk: str) -> bool:
        """
        Returns True if the chunk semantically matches an existing primitive.
        Uses SemanticRouter — pure embedding similarity, no LLM involved.
        result.name is None when the best score is below the route threshold,
        which acts as the OOS (out-of-scope) detection mechanism.
        """
        result = self.semantic_router(chunk)
        covered = result.name is not None

        logger.info(
            "'%s' → matched: '%s' → %s",
            chunk,
            result.name or "NONE",
            "COVERED" if covered else "NOT COVERED",
        )
        return covered