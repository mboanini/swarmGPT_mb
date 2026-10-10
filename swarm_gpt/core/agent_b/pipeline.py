import logging
from pathlib import Path

from swarm_gpt.core.agent_b.code_review import CodeReview
from swarm_gpt.core.agent_b.debug_function import DebugFunction
from swarm_gpt.core.agent_b.describe_primitive import DescribePrimitive
from swarm_gpt.core.agent_b.design_function import DesignFunction
from swarm_gpt.core.agent_b.function_node import FunctionNode
from swarm_gpt.core.agent_b.grammar_check import check as grammar_check, write_staging_file
from swarm_gpt.core.agent_b.mechanical_check import check as mechanical_check
from swarm_gpt.core.agent_b.runtime_cases import prepare_test_params, check_cases
from swarm_gpt.core.agent_b.validate_description import ValidateNameDesc
from swarm_gpt.core.agent_b.write_function import WriteFunction

logger = logging.getLogger(__name__)

_STAGING_DIR     = Path(__file__).resolve().parent / "staging"
_MAX_DEBUG_CALLS = 3


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

        node = FunctionNode(name=name, description=description, user_command=cmd)
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

        # --- Certification: StaticCheck, then RuntimeCheck; Debug repairs and we start over ---
        test_params = prepare_test_params(node.definition, node.n_args)  # inferred once, reused
        if test_params is None:
            raise RuntimeError(f"Unable to prepare runtime test parameters for '{name}'")
        debug_context = f"Description: {node.description}\n\nInterface:\n{node.definition}"

        for debug_calls in range(_MAX_DEBUG_CALLS + 1):
            write_staging_file(staging_path, node.body)
            errors = grammar_check(staging_path) + mechanical_check(node.body, name, node.definition)
            if not errors:
                errors = check_cases(node.body, name, test_params)
            if not errors:
                logger.info(f"Certified: {name} (n_args={node.n_args}, debug calls={debug_calls})")
                return node

            logger.warning(f"Certification attempt {debug_calls + 1}: {len(errors)} error(s)")
            for e in errors:
                logger.warning(f"  {e}")
            if debug_calls == _MAX_DEBUG_CALLS:
                break
            self._debugger.setup(node)
            self._debugger.set_errors(errors, context=debug_context)
            self._debugger.run()

        raise RuntimeError(
            f"Certification failed for '{name}' after {_MAX_DEBUG_CALLS} Debug calls:\n"
            + "\n".join(f"- {e}" for e in errors)
        )
