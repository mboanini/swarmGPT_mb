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

_STAGING_DIR  = Path(__file__).resolve().parent / "staging"
_MAX_REPAIRS = 3

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

    def _static_check(self, node: FunctionNode, staging_path: Path,) -> list[str]:
        write_staging_file(staging_path, node.body)

        return (grammar_check(staging_path) + mechanical_check(node.body, node.name, definition=node.definition))

    def _certify_static(
        self,
        node: FunctionNode,
        staging_path: Path,
    ) -> None:

        for repair_count in range(_MAX_REPAIRS + 1):
            static_errors = self._static_check(node, staging_path)

            if not static_errors:
                logger.info(
                    "Static certification passed after %d repair(s)",
                    repair_count,
                )
                return

            logger.warning(
                "Static certification failed after %d repair(s): %d error(s)",
                repair_count,
                len(static_errors),
            )

            for error in static_errors:
                logger.warning("  %s", error)

            if repair_count == _MAX_REPAIRS:
                raise RuntimeError(
                    f"Static certification failed for primitive "
                    f"'{node.name}' after {_MAX_REPAIRS} repair attempts:\n"
                    + "\n".join(f"- {error}" for error in static_errors)
                )

            self._debugger.setup(node)
            self._debugger.set_errors(
                static_errors,
                context=(
                    "The following Design interface is authoritative and immutable.\n"
                    "Preserve its # n_args comment, function signature, docstring, "
                    "and parameter names/order exactly.\n\n"
                    f"{node.definition}"
                ),
            )
            self._debugger.run()

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

        # --- Static Check loop (pylint + mechanical AST checks) ---
        self._certify_static(node, staging_path)

        # --- RuntimeCheck loop ---
        inferred = prepare_test_params(
            function_definition=node.definition,
            n_args=node.n_args,
        )

        if inferred is None:
            raise RuntimeError(
                f"Unable to prepare runtime test parameters for {node.name}"
            )

        runtime_context = (
            f"Description: {node.description}\n\n"
            f"Interface:\n{node.definition}"
        )

        for repair_count in range(_MAX_REPAIRS + 1):
            runtime_errors = check_cases(
                body=node.body,
                func_name=node.name,
                inferred=inferred,
            )

            if not runtime_errors:
                break

            if repair_count == _MAX_REPAIRS:
                raise RuntimeError(
                    f"Runtime certification failed for {node.name}: "
                    + "; ".join(runtime_errors)
                )

            self._debugger.setup(node)
            self._debugger.set_errors(
                runtime_errors,
                context=runtime_context,
            )
            self._debugger.run()

            # Every runtime repair must pass static certification again
            self._certify_static(node, staging_path)

        logger.info(f"Done: {name} (n_args={node.n_args})")
        return node
