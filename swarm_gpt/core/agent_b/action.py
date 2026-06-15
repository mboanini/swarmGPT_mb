from tenacity import retry, stop_after_attempt, wait_random_exponential

from swarm_gpt.core._llm_client import client
from swarm_gpt.core.agent_b.function_node import FunctionNode


class ActionNode:
    def __init__(self, model: str = "gpt-4o"):
        self.prompt = None    # formatted prompt string, filled by _build_prompt()
        self._node  = None    # FunctionNode, set via setup()
        self._model = model

    def setup(self, node: FunctionNode):
        self._node = node

    def run(self):
        self._build_prompt()
        response = self._call_llm()
        self._process_response(response)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10))
    def _call_llm(self) -> str:
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": self.prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def _build_prompt(self):
        raise NotImplementedError

    def _process_response(self, response: str) -> None:
        raise NotImplementedError
