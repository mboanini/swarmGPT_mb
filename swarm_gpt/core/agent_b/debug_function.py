"""DebugFunction: fix pylint or runtime errors in a generated motion primitive."""
from swarm_gpt.core.agent_b.action import ActionNode
from swarm_gpt.core.agent_b.function_node import State
from swarm_gpt.core.agent_b.parser import FunctionParser, parse_text


class DebugFunction(ActionNode):
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
        self._errors: list[str] = []
        self._context: str | None = None

    def set_errors(self, errors: list[str], context: str | None = None) -> None:
        self._errors = errors
        self._context = context

    def _build_prompt(self) -> None:
        error_block = "\n".join(f"  - {e}" for e in self._errors)
        context_block = f"\n\n## Context\n{self._context}" if self._context else ""
        self.prompt = (
            "The following Python function has errors. Fix them and return the corrected function. "
            "Keep the same name, signature and # n_args comment, and unpack `params` with the same "
            "names, in the same order and with the same meaning as the Interface."
            f"{context_block}\n\n"
            f"```python\n{self._node.body}\n```\n\n"
            f"Errors:\n{error_block}\n\n"
            "Return only the corrected function inside a ```python``` block."
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
