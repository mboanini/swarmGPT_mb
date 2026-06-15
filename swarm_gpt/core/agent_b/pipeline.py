import logging

from swarm_gpt.core.agent_b.describe_primitive import DescribePrimitive
from swarm_gpt.core.agent_b.design_function import DesignFunction
from swarm_gpt.core.agent_b.function_node import FunctionNode
from swarm_gpt.core.agent_b.write_function import WriteFunction

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, model: str = "gpt-4o"):
        self._describer = DescribePrimitive(model=model)
        self._designer  = DesignFunction(model=model)
        self._writer    = WriteFunction(model=model)

    def run(self, user_commands: list[str]) -> list[FunctionNode]:
        nodes = []
        for cmd in user_commands:
            logger.info(f"Describe stage: '{cmd}'")
            name, description = self._describer.run(cmd)

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
