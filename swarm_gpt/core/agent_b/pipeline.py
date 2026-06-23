import logging

from swarm_gpt.core.agent_b.describe_primitive import DescribePrimitive
from swarm_gpt.core.agent_b.design_function import DesignFunction
from swarm_gpt.core.agent_b.function_node import FunctionNode
from swarm_gpt.core.agent_b.write_function import WriteFunction
from swarm_gpt.core.agent_b.validate_description import ValidateNameDesc

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, model: str = "gpt-4o"):
        self._describer = DescribePrimitive(model=model)
        self._name_desc_validator = ValidateNameDesc()
        self._designer  = DesignFunction(model=model)
        self._writer    = WriteFunction(model=model)

    def run(self, user_commands: list[str]) -> list[FunctionNode]:
        nodes = []
        for cmd in user_commands:
            logger.info(f"Describe stage: '{cmd}'")
            name, description = self._describer.run(cmd)
            is_valid, errors = self._name_desc_validator.validate(name, description)
            if not is_valid:
                raise ValueError(f"Invalid primitive: {errors}")
            else:
                print("OK Validator")

            node = FunctionNode(name=name, description=description)

            logger.info(f"Design stage: {name}")
            self._designer.setup(node)
            self._designer.run()

            logger.info(f"Write stage: {name}")
            self._writer.setup(node)
            self._writer.run()

            logger.info(f"Done: {name} (n_args={node.n_args})")
            nodes.append(node)

        return nodes
