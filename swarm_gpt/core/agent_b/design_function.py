from swarm_gpt.core.agent_b.action import ActionNode
from swarm_gpt.core.agent_b.function_node import FunctionNode, State
from swarm_gpt.core.agent_b.parser import FunctionParser, parse_text
from swarm_gpt.core.agent_b.prompt import (
    CONSTRAINTS_TEXT,
    DESIGN_PROMPT,
    ENV_DES,
    FEW_SHOT_EXAMPLES,
    ROBOT_API_DESC,
    TASK_DES,
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
        self._node.definition = parser.function_definition
        self._node.n_args     = parser.n_args
        self._node.state      = State.DESIGNED
