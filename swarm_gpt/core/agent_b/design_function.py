import ast

from swarm_gpt.core.agent_b.action import ActionNode
from swarm_gpt.core.agent_b.function_node import FunctionNode, State
from swarm_gpt.core.agent_b.parser import FunctionParser, param_names, parse_text
from swarm_gpt.core.agent_b.prompt import (
    CONSTRAINTS_TEXT,
    DESIGN_PROMPT,
    ENV_DES,
    FEW_SHOT_EXAMPLES,
    ROBOT_API_DESC,
    TASK_DES,
)

_SIGNATURE = ["params", "swarm_pos", "tstart", "tend", "limits"]


def _check_interface(definition: str, name: str, n_args: int) -> None:
    """Raise ValueError if the Design interface is unusable. Design errors are not sent to Debug."""
    function = next(
        n for n in ast.parse(definition).body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    if [a.arg for a in function.args.args] != _SIGNATURE:
        raise ValueError(f"Design '{name}': signature must be {name}({', '.join(_SIGNATURE)})")
    if n_args < 1:
        raise ValueError(f"Design '{name}': '# n_args: N' comment is missing or invalid")
    try:
        names = param_names(definition, name)
    except ValueError as exc:
        raise ValueError(f"Design '{name}': {exc}") from exc
    if len(names) != n_args:
        raise ValueError(
            f"Design '{name}': # n_args: {n_args} but docstring declares {len(names)} "
            f"parameter(s) {tuple(names)}"
        )


class DesignFunction(ActionNode):
    def _build_prompt(self):
        self.prompt = DESIGN_PROMPT.format(
            task_des        = TASK_DES,
            env_des         = ENV_DES,
            function_name   = self._node.name,
            function_des    = self._node.description,
            robot_api       = ROBOT_API_DESC,
            other_functions = FEW_SHOT_EXAMPLES,
            constraints     = CONSTRAINTS_TEXT,
        )

    def _process_response(self, response: str):
        code = parse_text(response)
        parser = FunctionParser()
        parser.parse(code)
        parser.check_function_name(self._node.name)
        _check_interface(parser.function_definition, self._node.name, parser.n_args)
        self._node.definition = parser.function_definition
        self._node.n_args     = parser.n_args
        self._node.state      = State.DESIGNED
