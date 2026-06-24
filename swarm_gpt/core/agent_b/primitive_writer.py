import json
from pathlib import Path

from swarm_gpt.core._llm_client import client
from swarm_gpt.core.agent_b.function_node import FunctionNode
from swarm_gpt.core.agent_b.prompt.generate_utterances_prompt import UTTERANCES_PROMPT

# insert in dict and append in motion_primitives.py
def write_to_file(nodes: list[FunctionNode], file_path: Path) -> None:
    content = file_path.read_text()
    for node in nodes:
        if f'\ndef {node.name}(' not in content:
            content = _insert_dict_entry(content, node.name, node.n_args)
        content = _replace_or_append_function(content, node.name, node.body)
    file_path.write_text(content)

# replace?
def _replace_or_append_function(content: str, name: str, body: str) -> str:
    """Replace an existing function definition in content, or append if not found."""
    marker = f'\ndef {name}('
    idx = content.find(marker)
    if idx == -1:
        return content.rstrip() + f'\n\n\n{body}\n'
    # find the next "def" or end of file
    next_def = content.find('\ndef ', idx + 1)
    if next_def == -1:
        return content[:idx] + f'\n\n\n{body}\n'
    return content[:idx] + f'\n\n\n{body}\n\n' + content[next_def + 1:]

def _insert_dict_entry(content: str, name: str, n_args: int) -> str:
    lines = content.split('\n')
    in_dict = False
    dict_end_line = -1
    for i, line in enumerate(lines):
        if 'motion_primitives = {' in line:
            in_dict = True
        elif in_dict and line.strip() == '}':
            dict_end_line = i
            break

    if dict_end_line < 0:
        raise ValueError("Could not find closing } of motion_primitives dict")

    lines.insert(dict_end_line, f'    "{name}": {{"n_args": {n_args}}},')
    return '\n'.join(lines)


# Semantic Router (Route) + primitives.yaml
def _generate_utterances(name: str, description: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": UTTERANCES_PROMPT},
            {"role": "user", "content": f"name: {name}\ndescription: {description}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("utterances", [])


def register_with_router(nodes: list[FunctionNode], router) -> None:
    for node in nodes:
        utterances = _generate_utterances(node.name, node.description)
        router.register_primitive(
            name=node.name,
            description=node.description,
            n_args=node.n_args,
            utterances=utterances,
        )
