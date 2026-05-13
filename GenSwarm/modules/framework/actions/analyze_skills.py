"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

from modules.file import logger
from modules.framework.action import ActionNode
from modules.framework.constraint import ConstraintPool
from modules.framework.code import FunctionTree
from modules.framework.parser import *
from modules.prompt import (
    ANALYZE_SKILL_PROMPT_TEMPLATE,
    FUNCTION_TEMPLATE,
    ALLOCATOR_TEMPLATE,
    ENV_DES,
    TASK_DES,
)
from modules.utils import root_manager, rich_print


class AnalyzeSkills(ActionNode):
    def __init__(self, next_text, node_name=""):
        super().__init__(next_text, node_name)
        self._constraint_pool: ConstraintPool = self.context.constraint_pool

    def _build_prompt(self):
        self.prompt = self.prompt.format(
            task_des=TASK_DES,
            instruction=self.context.command,
            local_api=self.context.local_robot_api
            + ALLOCATOR_TEMPLATE.format(template="Temporarily unknown"),
            global_api=self.context.global_robot_api,
            env_des=ENV_DES,
            constraints=str(self._constraint_pool),
            output_template=FUNCTION_TEMPLATE,
        )
        self.set_logging_text(f"Analyzing skills")

    async def _process_response(self, response: str) -> str:
        content = parse_text(response, "json")
        functions = eval(content)["functions"]
        global_functions = []
        local_functions = []
        for function in functions:
            if function["scope"] == "global":
                global_functions.append(function)
            else:
                local_functions.append(function)
        self.context.global_skill_tree.init_functions(global_functions)
        self.context.local_skill_tree.init_functions(local_functions)
        if len(global_functions) == 0:
            logger.log("No global functions detected,generate local skills", "warning")
            self.context.scoop = "local"
            from modules.framework.actions import GenerateFunctions

            self._next = GenerateFunctions()

        self._constraint_pool.check_constraints_satisfaction()
        logger.log(f"Analyze Functions Success", "success")
        return response

    def _display(self):
        content = ""

        functions_nodes = list(self.context.global_skill_tree.nodes) + list(
            self.context.local_skill_tree.nodes
        )
        for index, function in enumerate(functions_nodes):
            content += f"[bold yellow]{index+1}. {function.name}[/bold yellow]\n"
            content += f"{function.description}\n"

        def print_tree(tree, content):
            layers = tree._layers
            for layer_index, layer in enumerate(layers):
                content += f"Layer {layer_index}:\n"
                for function_node in layer:
                    content += f"  - {function_node.name}\n"
            return content

        global_content = print_tree(self.context.global_skill_tree, "")
        local_content = print_tree(self.context.local_skill_tree, "")
        rich_print("Step 2: Analyze Skills", content)
        rich_print("Step 2: Analyze Skills - Global Skill Graph", global_content)
        rich_print("Step 2: Analyze Skills - Local Skill Graph", local_content)


if __name__ == "__main__":
    import asyncio

    function_analyser = AnalyzeSkills("analyze constraints")
    path = "../../../workspace/test"
    root_manager.update_root("../../../workspace/test")

    function_analyser.context.load_from_file(f"{path}/constraint.pkl")
    asyncio.run(function_analyser.run())
    function_analyser.context.save_to_file(f"{path}/analyze_functions.pkl")
