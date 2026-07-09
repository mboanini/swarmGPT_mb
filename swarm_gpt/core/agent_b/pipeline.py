import logging
from pathlib import Path

from swarm_gpt.core.agent_b.code_review import CodeReview
from swarm_gpt.core.agent_b.debug_function import DebugFunction
from swarm_gpt.core.agent_b.describe_primitive import DescribePrimitive
from swarm_gpt.core.agent_b.design_function import DesignFunction
from swarm_gpt.core.agent_b.function_node import FunctionNode
from swarm_gpt.core.agent_b.grammar_check import check as grammar_check, write_staging_file
from swarm_gpt.core.agent_b.infer_test_params import infer as infer_test_params
from swarm_gpt.core.agent_b.mechanical_check import check as mechanical_check
from swarm_gpt.core.agent_b.runtime_check import run as runtime_check
from swarm_gpt.core.agent_b.validate_description import ValidateNameDesc
from swarm_gpt.core.agent_b.write_function import WriteFunction

logger = logging.getLogger(__name__)

_STAGING_DIR  = Path(__file__).resolve().parent / "staging"
_MAX_ATTEMPTS = 3


class Pipeline:
    def __init__(self, model: str = "gpt-4o"):
        self._describer           = DescribePrimitive(model=model)
        self._name_desc_validator = ValidateNameDesc()
        self._designer            = DesignFunction(model=model)
        self._writer              = WriteFunction(model=model)
        self._reviewer            = CodeReview(model=model)
        self._debugger            = DebugFunction(model=model)

    def run(self, user_commands: list[str]) -> list[FunctionNode]:
        nodes = []
        for cmd in user_commands:
            node = self._run_one(cmd)
            nodes.append(node)
        return nodes

    def _run_one(self, cmd: str) -> FunctionNode:
        # --- Describe ---
        logger.info(f"Describe stage: '{cmd}'")
        name, description = self._describer.run(cmd)
        is_valid, errors = self._name_desc_validator.validate(name, description)
        if not is_valid:
            raise ValueError(f"Invalid primitive: {errors}")
        print("OK Validator")

        node         = FunctionNode(name=name, description=description, user_command=cmd)
        staging_path = _STAGING_DIR / f"{name}.py"

        # --- Design ---
        logger.info(f"Design stage: {name}")
        self._designer.setup(node)
        self._designer.run()

        # --- Write ---
        logger.info(f"Write stage: {name}")
        self._writer.setup(node)
        self._writer.run()

        # --- CodeReview ---
        logger.info(f"CodeReview stage: {name}")
        self._reviewer.setup(node)
        self._reviewer.run()

        # --- Static Check loop (pylint + mechanical AST checks) ---
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            write_staging_file(staging_path, node.body)
            static_errors = grammar_check(staging_path) + mechanical_check(node.body, node.name)
            if not static_errors:
                logger.info(f"StaticCheck OK (attempt {attempt})")
                break
            logger.warning(f"StaticCheck attempt {attempt}: {len(static_errors)} error(s)")
            for e in static_errors:
                logger.warning(f"  {e}")
            if attempt == _MAX_ATTEMPTS:
                logger.error("StaticCheck: max attempts reached, proceeding with current code")
                break
            self._debugger.setup(node)
            self._debugger.set_errors(static_errors)
            self._debugger.run()

        # --- RuntimeCheck loop ---
        test_params = infer_test_params(node.definition)
        if test_params is not None:
            runtime_context = (
                f"Description: {node.description}\n\n"
                f"Interface:\n{node.definition}"
            )
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                runtime_errors = runtime_check(node.body, node.name, test_params)
                if not runtime_errors:
                    logger.info(f"RuntimeCheck OK (attempt {attempt})")
                    break
                logger.warning(f"RuntimeCheck attempt {attempt}: {len(runtime_errors)} error(s)")
                for e in runtime_errors:
                    logger.warning(f"  {e}")
                if attempt == _MAX_ATTEMPTS:
                    logger.error("RuntimeCheck: max attempts reached, proceeding with current code")
                    break
                self._debugger.setup(node)
                self._debugger.set_errors(runtime_errors, context=runtime_context)
                self._debugger.run()
        else:
            logger.warning(f"RuntimeCheck skipped: infer_test_params failed for {name}")

        logger.info(f"Done: {name} (n_args={node.n_args})")
        return node
