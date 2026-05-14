"""SwarmOrchestrator: agent-based decision flow for Crazyflie swarm control.

Flow
----
NL command
    └─► Router Agent
            ├─► [Feasible]   existing primitive(s) found
            │       └─► Choreographer.generate_choreography()
            └─► [New]        behaviour requires a new primitive
                    └─► GenSwarm Coder (offline, centralised)
                            ├─► GenSwarmBridge.synthesize()
                            ├─► DynamicLibrary.register()  (persist + hot-load)
                            └─► Choreographer.generate_choreography()  (reuses new primitive)
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from swarm_gpt.core.choreographer import Choreographer

from swarmgenpt.dynamic_library import DynamicLibrary
from swarmgenpt.genswarm_bridge import GenSwarmBridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class RouteAction(Enum):
    EXISTING = auto()   # handled by current primitives
    GENERATE = auto()   # requires a new primitive


@dataclass
class RouteDecision:
    action: RouteAction
    reason: str
    # EXISTING path
    primitives: list[str] = field(default_factory=list)
    # GENERATE path
    fn_name: str = ""
    fn_description: str = ""
    n_args_hint: int = 0


@dataclass
class ExecutionResult:
    waypoints: dict          # {"time": NDArray, "pos": NDArray, ...}
    route: RouteDecision
    generated_fn: str = ""   # non-empty when a new primitive was synthesised


# ---------------------------------------------------------------------------
# Router Agent prompt
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = textwrap.dedent("""\
    You are the Router Agent of a Crazyflie drone-swarm choreographer.
    Given a user command and the list of available motion primitives, decide:

    OPTION A — the command can be implemented with EXISTING primitives.
    OPTION B — the command requires a NEW primitive (a novel movement pattern
               not achievable by composing existing ones).

    ## Available motion primitives
    {primitive_list}

    ## Output format — respond with a single JSON object, nothing else:

    Option A:
    {{
      "action": "existing",
      "primitives": ["primitive_call_1(args)", "primitive_call_2(args)"],
      "reason": "<one sentence>"
    }}

    Option B:
    {{
      "action": "generate",
      "fn_name": "<snake_case_identifier>",
      "fn_description": "<precise natural-language spec for the code generator>",
      "n_args_hint": <estimated number of numeric params>,
      "reason": "<one sentence>"
    }}

    ## Rules
    - Prefer EXISTING if any combination of current primitives approximates the request.
    - Choose GENERATE only when genuinely new geometry or dynamics is required.
    - fn_name must be a valid Python identifier in snake_case.
    - fn_description must be self-contained: mention shape, timing, spacing requirements.
    - Respond with JSON only — no markdown fences, no prose.
""")

_ROUTER_USER = textwrap.dedent("""\
    Number of drones: {n_drones}
    Spatial limits (m): lower={lim_lower}, upper={lim_upper}

    User command:
    {command}
""")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SwarmOrchestrator:
    """Central router + code-generation pipeline for the Crazyflie swarm.

    Args:
        choreographer: An initialised Choreographer instance (owns drone config
                       and LLM connection).
        dynamic_lib:   Optional pre-existing DynamicLibrary.  A new one is
                       created if not provided.
        bridge:        Optional pre-existing GenSwarmBridge.  A new one is
                       created if not provided.
        router_model:  Override the model used by the Router Agent.
    """

    def __init__(
        self,
        choreographer: "Choreographer",
        dynamic_lib: DynamicLibrary | None = None,
        bridge: GenSwarmBridge | None = None,
        router_model: str | None = None,
    ) -> None:
        self._choreo = choreographer
        self._library = dynamic_lib or DynamicLibrary()
        self._bridge = bridge or GenSwarmBridge()
        self._router_model = router_model

        # Hot-load any previously generated primitives so they are available
        # immediately without a synthesis round-trip.
        n = self._library.load_all_into_module()
        if n:
            logger.info("Loaded %d previously generated primitive(s)", n)

        self._router_client, self._router_backend, self._router_model_id = (
            self._init_router_client(router_model)
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, nl_command: str) -> ExecutionResult:
        """Full pipeline: route → (synthesise if needed) → generate waypoints.

        Args:
            nl_command: Free-text instruction from the user.

        Returns:
            ExecutionResult with waypoints ready for the drone controller.
        """
        decision = self.route(nl_command)
        logger.info("Router decision: %s — %s", decision.action.name, decision.reason)

        generated_fn = ""

        if decision.action == RouteAction.GENERATE:
            generated_fn = self._synthesise_and_register(decision)
            # Rewrite the command so the Choreographer uses the new primitive
            nl_command = self._augment_command_with_new_primitive(
                nl_command, decision
            )

        # Generate waypoints through the standard Choreographer pipeline
        prompt = self._choreo.format_initial_prompt(nl_command)
        llm_response = self._choreo.generate_choreography(prompt)
        waypoints = self._choreo.response2waypoints(llm_response)

        return ExecutionResult(
            waypoints=waypoints,
            route=decision,
            generated_fn=generated_fn,
        )

    def route(self, nl_command: str) -> RouteDecision:
        """Router Agent: classify the command and decide execution path.

        Args:
            nl_command: Free-text instruction from the user.

        Returns:
            RouteDecision describing the chosen path.
        """
        from swarm_gpt.core.motion_primitives import motion_primitives

        primitive_list = "\n".join(
            f"  - {name}  (n_args={meta['n_args']})"
            for name, meta in motion_primitives.items()
        )
        system_prompt = _ROUTER_SYSTEM.format(primitive_list=primitive_list)
        user_prompt = _ROUTER_USER.format(
            n_drones=self._choreo.num_drones,
            lim_lower=list(self._choreo.lim_lower),
            lim_upper=list(self._choreo.lim_upper),
            command=nl_command,
        )

        raw = self._call_router(system_prompt, user_prompt)
        return self._parse_router_response(raw)

    # ------------------------------------------------------------------
    # Synthesis pipeline
    # ------------------------------------------------------------------

    def _synthesise_and_register(self, decision: RouteDecision) -> str:
        """Call GenSwarmBridge, validate output, persist and hot-load."""
        limits = {
            "lower": self._choreo.lim_lower,
            "upper": self._choreo.lim_upper,
        }

        if self._library.exists(decision.fn_name):
            logger.info(
                "Primitive '%s' already in library — skipping synthesis",
                decision.fn_name,
            )
            return decision.fn_name

        result = self._bridge.synthesize(
            fn_name=decision.fn_name,
            description=decision.fn_description,
            n_drones=self._choreo.num_drones,
            limits=limits,
        )

        self._library.register(
            fn_name=result.fn_name,
            n_args=result.n_args,
            description=result.description,
            source_code=result.source_code,
        )
        logger.info(
            "New primitive '%s' registered (n_args=%d)", result.fn_name, result.n_args
        )
        return result.fn_name

    @staticmethod
    def _augment_command_with_new_primitive(
        original_command: str, decision: RouteDecision
    ) -> str:
        """Append a hint so the Choreographer uses the newly registered primitive."""
        return (
            f"{original_command}\n\n"
            f"[System hint] A new motion primitive '{decision.fn_name}' has just been "
            f"added to the library. Use it for the movement described above."
        )

    # ------------------------------------------------------------------
    # Router LLM calls
    # ------------------------------------------------------------------

    @staticmethod
    def _init_router_client(model_id: str | None):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            mid = model_id or "claude-haiku-4-5-20251001"   # fast, cheap for routing
            return client, "anthropic", mid

        if openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            mid = model_id or "gpt-4o-mini"
            return client, "openai", mid

        raise EnvironmentError(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set."
        )

    def _call_router(self, system_prompt: str, user_prompt: str) -> str:
        if self._router_backend == "anthropic":
            response = self._router_client.messages.create(
                model=self._router_model_id,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        response = self._router_client.chat.completions.create(
            model=self._router_model_id,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_router_response(raw: str) -> RouteDecision:
        """Parse the Router Agent JSON output into a RouteDecision."""
        # Strip markdown fences if the model included them despite instructions
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Router Agent returned invalid JSON: {exc}\nRaw:\n{raw}"
            ) from exc

        action_str = data.get("action", "").lower()
        if action_str == "existing":
            return RouteDecision(
                action=RouteAction.EXISTING,
                reason=data.get("reason", ""),
                primitives=data.get("primitives", []),
            )
        elif action_str == "generate":
            return RouteDecision(
                action=RouteAction.GENERATE,
                reason=data.get("reason", ""),
                fn_name=data.get("fn_name", "custom_move"),
                fn_description=data.get("fn_description", ""),
                n_args_hint=int(data.get("n_args_hint", 2)),
            )
        else:
            raise ValueError(
                f"Router Agent returned unknown action '{action_str}'. Raw:\n{raw}"
            )


# ---------------------------------------------------------------------------
# Example: wrapping a genSwarm-generated function for the swarmGPT interface
# ---------------------------------------------------------------------------

# The block below is NOT imported at runtime.  It illustrates how a function
# synthesised by GenSwarmBridge would look once it arrives in the
# generated_library/ directory and gets injected into motion_primitives.

_EXAMPLE_GENERATED_FUNCTION = textwrap.dedent("""\
    # PARAMS_SCHEMA: n_args=4
    def lissajous_formation(
        params: tuple,
        swarm_pos: NDArray,
        tstart: float,
        tend: float,
        limits: dict,
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        \"\"\"Arrange drones on a 3-D Lissajous curve at fixed height.\"\"\"
        steps, height, freq_ratio, phase_shift = params
        # steps: int      — number of waypoints
        # height: int     — altitude in cm
        # freq_ratio: int — y-frequency relative to x (e.g. 3 means ωy = 3·ωx)
        # phase_shift: int — phase offset in degrees

        n_drones = swarm_pos.shape[0]
        lim_lower, lim_upper = limits["lower"], limits["upper"]
        steps = max(int(steps), int(tend - tstart))
        height = int(np.clip(height, lim_lower[2] * 100, lim_upper[2] * 100))
        phase = np.deg2rad(float(phase_shift))

        # Scale to fit inside the permitted XY envelope (in cm)
        x_amp = min((lim_upper[0] - lim_lower[0]) * 100 / 2, 150)
        y_amp = min((lim_upper[1] - lim_lower[1]) * 100 / 2, 150)

        # Distribute n_drones evenly around the Lissajous parameter space
        t_params = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
        x = x_amp * np.sin(t_params)
        y = y_amp * np.sin(freq_ratio * t_params + phase)
        des_pos = np.stack([x, y, np.full(n_drones, height)], axis=1)

        # Enforce minimum 60 cm spacing with Hungarian assignment
        assignment = _assign_positions(swarm_pos, des_pos)
        des_pos = des_pos[assignment]

        # Enforce max velocity: interpolate linearly from current to desired
        max_vel_per_step = 100.0 * (tend - tstart) / steps
        waypoints = {}
        for k, t in enumerate(np.linspace(tstart, tend, steps + 1)[1:]):
            alpha = (k + 1) / steps
            pos = swarm_pos + alpha * np.clip(
                des_pos - swarm_pos,
                -max_vel_per_step,
                max_vel_per_step,
            )
            pos[:, 2] = height
            waypoints[t] = {i: p.copy() for i, p in enumerate(pos)}

        return des_pos, waypoints
""")
