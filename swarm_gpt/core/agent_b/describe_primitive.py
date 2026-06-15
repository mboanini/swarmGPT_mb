import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from swarm_gpt.core._llm_client import client
from swarm_gpt.core.motion_primitives import motion_primitives

_DESCRIBE_PROMPT = """
You are helping design a new motion primitive for a Crazyflie 2.1 drone swarm controller.

Given a user command, extract:
1. A snake_case function name for the new motion primitive
2. A 1-2 sentence formal description of what the primitive does, suitable for guiding code generation

Existing primitive names (do NOT reuse these):
{existing_names}

User command: "{user_command}"

Respond with a JSON object only:
{{"name": "snake_case_name", "description": "Formal description..."}}
""".strip()


class DescribePrimitive:
    def __init__(self, model: str = "gpt-4o"):
        self._model = model

    def run(self, user_command: str) -> tuple[str, str]:
        prompt = _DESCRIBE_PROMPT.format(
            existing_names=", ".join(motion_primitives.keys()),
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
