import json
from pathlib import Path

import yaml
from tenacity import retry, stop_after_attempt, wait_random_exponential

from swarm_gpt.core._llm_client import client
from swarm_gpt.core.agent_b.prompt.func_description_prompt import DESCRIBE_PROMPT

_PRIMITIVES_PATH = Path(__file__).resolve().parents[2] / "data/primitives.yaml"


class DescribePrimitive:
    def __init__(self, model: str = "gpt-4o"):
        self._model = model

    def run(self, user_command: str) -> tuple[str, str]:
        with open(_PRIMITIVES_PATH) as f:
            primitives = yaml.safe_load(f)
        existing_primitives = "\n".join(
            f"- {name}: {data['description']}" for name, data in primitives.items()
        )
        prompt = DESCRIBE_PROMPT.format(
            existing_primitives=existing_primitives,
            user_command=user_command,
        )
        data = self._call_llm(prompt)
        return data["name"], data["description"]

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10))
    def _call_llm(self, prompt: str) -> dict:
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        return json.loads(response.choices[0].message.content)
