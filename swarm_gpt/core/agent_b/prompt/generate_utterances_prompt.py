UTTERANCES_PROMPT = """
You are helping describe a new motion primitive for a Crazyflie 2.1 drone swarm controller.

Given the name and description of a motion primitive just added, generate 5-10 different ways
a user could request this action in natural language.

Respond with a JSON object only:
{{
    "utterances": [
        "utterance 1",
        "utterance 2",
        ...
    ]
}}
""".strip()
