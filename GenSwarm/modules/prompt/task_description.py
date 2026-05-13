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

TASK_DES = """
There are some mobile ground robots that can control their own speed, acquire their own location, and sense the positions and speeds of other robots within their field of view.
There is a control center that can gather global information about all the robots and is responsible for task allocation, assisting the robots in performing collaborative tasks.
The control center only performs task allocation at the beginning, and thereafter the robots' autonomous movement is entirely dependent on themselves.
The allocator needs to consider the initial state information of all robots and assign corresponding sub-goals to each robot based on the task objectives.
The designed allocation algorithm must provide an optimal allocation strategy while avoiding conflicts between robots' goals.
The control center's allocator runs only once at the start of the task and is responsible for assigning complete tasks to each robot. After that, the robots should be able to independently complete these tasks without any subsequent communication with the control center.
Not all tasks require allocation by the control center. Robots obtain assigned sub-tasks through APIs; if no corresponding API is provided, the task can be completed independently by each robot.
Currently, multiple AI assistants are collaborating step-by-step to write code that runs on both the control center and the ground robots.
You are one of these assistants, and you need to understand this context and carry out your work accordingly.
""".strip()
