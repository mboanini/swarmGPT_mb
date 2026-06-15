DESCRIBE_PROMPT = """
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