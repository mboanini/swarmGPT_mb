DESCRIBE_PROMPT = """
You are helping design a new motion primitive for a Crazyflie 2.1 drone swarm controller.

Given a user command, extract:
1. A snake_case function name for the new motion primitive
2. A 1-2 sentence description of what the primitive does, in the same style
   as the existing primitives below: describe the concrete geometric/mechanical
   behavior (what shape is formed, how it's computed, how it moves) rather than
   a vague label. Do not just restate the user command.

Existing primitives (do NOT reuse these names), also given as a style
reference for how to write the description:
{existing_primitives}

User command: "{user_command}"

Respond with a JSON object only:
{{"name": "snake_case_name", "description": "Formal description..."}}
""".strip()