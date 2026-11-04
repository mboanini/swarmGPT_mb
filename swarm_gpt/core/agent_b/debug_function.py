"""DebugFunction: fix pylint or runtime errors in a generated motion primitive."""
from swarm_gpt.core.agent_b.action import ActionNode
from swarm_gpt.core.agent_b.function_node import State
from swarm_gpt.core.agent_b.parser import FunctionParser, parse_text
from swarm_gpt.core.agent_b.prompt import (
    CONSTRAINTS_TEXT,
    DEBUG_PROMPT,
    ENV_DES,
    IMPLEMENTATION_RULES,
    ROBOT_API_DESC,
)


class DebugFunction(ActionNode):
    def __init__(self, model: str = "gpt-6-luna"):
        super().__init__(model)
        self._errors: list[str] = []
        self._context: str | None = None

    def set_errors(self, errors: list[str], context: str | None = None) -> None:
        self._errors = errors
        self._context = context

    def _build_prompt(self) -> None:
        error_block = "\n".join(f"  - {e}" for e in self._errors)
        context_block = f"\n\n## Context\n{self._context}" if self._context else ""
        # same code rules and constraints as the Write stage
        self.prompt = DEBUG_PROMPT.format(
            context_block        = context_block,
            function_body        = self._node.body,
            error_block          = error_block,
            robot_api            = ROBOT_API_DESC,
            implementation_rules = IMPLEMENTATION_RULES.format(env_des=ENV_DES),
            constraints          = CONSTRAINTS_TEXT,
        )

    def _process_response(self, response: str) -> None:
        try:
            code = parse_text(response)
            parser = FunctionParser()
            parser.parse(code)
            parser.check_function_name(self._node.name)
            self._node.body   = parser.function_definition
            self._node.state  = State.IMPLEMENTED
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"DebugFunction: could not parse corrected code ({exc}), keeping original"
            )
