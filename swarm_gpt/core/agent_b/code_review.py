"""CodeReview: two-stage LLM review of a generated motion primitive.

Stage 0 — BehavioralComparison: checks semantic consistency with the user
           intent. Output: SATISFIED or NOT_SATISFIED + mismatch list.
Stage 1 — CodeFix: triggered only on NOT_SATISFIED. Receives the mismatch
           list from Stage 0 and returns a corrected function body.
"""
import logging

from swarm_gpt.core.agent_b.action import ActionNode
from swarm_gpt.core.agent_b.function_node import State
from swarm_gpt.core.agent_b.parser import FunctionParser, parse_text
from swarm_gpt.core.agent_b.prompt import (
    BEHAVIORAL_COMPARISON_PROMPT,
    CODE_REVIEW_PROMPT,
    CONSTRAINTS_TEXT,
    ENV_DES,
    FEW_SHOT_EXAMPLES,
    ROBOT_API_DESC,
    TASK_DES,
)

logger = logging.getLogger(__name__)

_NOT_SATISFIED_PREFIX = "NOT_SATISFIED"


class CodeReview(ActionNode):
    def run(self) -> None:
        # --- Stage 0: behavioral comparison ---
        self.prompt = BEHAVIORAL_COMPARISON_PROMPT.format(
            task_des        = TASK_DES,
            user_command    = self._node.user_command,
            function_name   = self._node.name,
            function_des    = self._node.description,
            function_body   = self._node.body,
            robot_api       = ROBOT_API_DESC,
            other_functions = FEW_SHOT_EXAMPLES,
        )
        response = self._call_llm()

        if not response.strip().startswith(_NOT_SATISFIED_PREFIX):
            logger.info("CodeReview/BehavioralComparison: SATISFIED — no changes")
            return

        mismatch_list = response.strip()[len(_NOT_SATISFIED_PREFIX):].lstrip(":").strip()
        logger.info(f"CodeReview/BehavioralComparison: NOT_SATISFIED — {mismatch_list[:120]}")

        # --- Stage 1: targeted fix ---
        self.prompt = CODE_REVIEW_PROMPT.format(
            task_des            = TASK_DES,
            env_des             = ENV_DES,
            user_command        = self._node.user_command,
            function_name       = self._node.name,
            function_des        = self._node.description,
            function_definition = self._node.definition,
            function_body       = self._node.body,
            robot_api           = ROBOT_API_DESC,
            other_functions     = FEW_SHOT_EXAMPLES,
            constraints         = CONSTRAINTS_TEXT,
            mismatch_list       = mismatch_list,
        )
        response = self._call_llm()
        self._process_response(response)

    def _build_prompt(self) -> None:
        pass  # prompts are built directly in run()

    def _process_response(self, response: str) -> None:
        try:
            code = parse_text(response)
            parser = FunctionParser()
            parser.parse(code)
            parser.check_function_name(self._node.name)
            self._node.body   = parser.function_definition
            self._node.state  = State.IMPLEMENTED
            logger.info("CodeReview/CodeFix: corrections applied")
        except Exception as exc:
            logger.warning(f"CodeReview/CodeFix: could not parse corrected code ({exc}), keeping original")
