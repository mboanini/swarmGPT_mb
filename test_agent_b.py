"""Manual end-to-end test for the Agent B pipeline.

Run with:
    OPENAI_API_KEY=sk-... python3 test_agent_b.py
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not set.")
    sys.exit(1)

from swarm_gpt.core.agent_b.pipeline import Pipeline

USER_COMMANDS = [
    "make the drones trace a figure eight path",
]

if __name__ == "__main__":
    pipeline = Pipeline(model="gpt-4o")
    print(f"\n--- Running pipeline for {USER_COMMANDS} ---\n")

    nodes = pipeline.run(USER_COMMANDS)

    for node in nodes:
        print(f"\n=== RESULT: {node.name} ===")
        print(f"state:  {node.state}")
        print(f"n_args: {node.n_args}")
        print(f"\n--- definition ---\n{node.definition}")
        print(f"\n--- body ---\n{node.body}")
