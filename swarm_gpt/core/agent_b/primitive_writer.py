import re
from pathlib import Path

import yaml

from swarm_gpt.core.agent_b.function_node import FunctionNode

_GENERATED_YAML_PATH = Path(__file__).resolve().parents[2] / "data/prompt_generated_primitives.yaml"

# insert into prompts_no_music.yaml
class _LiteralStr(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


class _LiteralDumper(yaml.Dumper):
    pass

# to mantain the original line spaces and breaks (no everything in a single line)
_LiteralDumper.add_representer(_LiteralStr, _literal_representer)

def write_descriptions_to_yaml(
    nodes: list[FunctionNode], yaml_path: Path = _GENERATED_YAML_PATH
) -> None:
    """Append newly generated primitive descriptions to prompt_generated_primitives.yaml.
    """
    # open yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    content: str = data.get("content") or ""

    for node in nodes:
        if f"  {node.name}(" in content:
            continue  # already present, skip to avoid conflicting duplicates
        entry = _format_entry(node)
        content = content.rstrip("\n") + "\n\n" + entry if content.strip() else entry

    data["content"] = _LiteralStr(content)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, Dumper=_LiteralDumper, allow_unicode=True, sort_keys=False)


def _format_entry(node: FunctionNode) -> str:
    """Format one primitive in prompts_no_music.yaml style."""
    defn = node.definition or ""
    param_names = _extract_param_names(defn, node.n_args)
    sig = f"{node.name}({', '.join(param_names)})"

    lines = [f"  {sig}"]
    lines.append(f"    - {node.description}")
    for pname, pdesc in _extract_param_descs(defn, set(param_names)):
        lines.append(f"    - {{{pname}}}: {pdesc}")
    return "\n".join(lines)


def _extract_param_names(defn: str, n_args: int) -> list[str]:
    """Extract param names from docstring. Tries inline format first, then per-line."""
    # inline: params: tuple[...] — (name1, name2, ...)
    m = re.search(r"params:\s*tuple\[.*?\]\s*[—\-]+\s*\(([\w,\s]+)\)", defn)
    if m:
        return [n.strip() for n in m.group(1).split(",")][:n_args]
    # per-line: "    name: type — description"
    names = re.findall(r"^\s{4,8}(\w+)\s*:\s*\S+\s*[—\-]", defn, re.MULTILINE)
    if names:
        return names[:n_args]
    return [f"p{i + 1}" for i in range(n_args)]


def _extract_param_descs(defn: str, param_names: set) -> list[tuple[str, str]]:
    """Extract per-param descriptions from the docstring params section."""
    m = re.search(r"params:\s*tuple.*?\n(.*?)(?=\s*swarm_pos:)", defn, re.DOTALL)
    if not m:
        return []
    results = []
    for line in m.group(1).splitlines():
        pm = re.match(r"\s+(\w+)\s*:\s*\S+\s*[—\-]+\s*(.+)", line)
        if pm and pm.group(1) in param_names:
            results.append((pm.group(1), pm.group(2).strip()))
    return results


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


def register_with_router(nodes: list[FunctionNode], router) -> None:
    for node in nodes:
        router.register_primitive(
            name=node.name,
            description=node.description,
            n_args=node.n_args,
        )
