WRITE_GLOBAL_FUNCTION_PROMPT_TEMPLATE = """
## Background:
{task_des}
## Role setting:
- Your task is to complete this predefined function according to the description.

## These are the environment description:
{env_des}

## These are the User original instructions:
{instruction}

## These are the available APIs (query and execution):
```python
{robot_api}
```

## These are the functions you can call directly even if they are not implemented yet:
```python
{other_functions}
```

## These are the constraints that need to be satisfied in the implementation of the function:
{constraints}

## Task
Complete the following function. The output TEXT format is as follows:
### Reasoning: (reason step by step about how to implement this function.)
### Code:
```python
import ...(if necessary)
{function_content}
    ...(function body, you need to complete it)
```

## Notes:
- All APIs (query and motion/execution) can be called directly without imports.
- Generate bug-free, directly invocable code according to Google's coding standards.
- The function must EXECUTE behavior: use query APIs to get state, compute targets with any
  needed algorithm, and call motion/execution APIs (move, form_circle, rotate, etc.) to
  carry out the movement. Do NOT merely return a dictionary of positions for robots to use later.
- A function returns None if it executes movement. It may return a computed value only if
  it is a pure helper whose result is consumed by a calling function that does the execution.
- Adjustable parameters should be taken as input parameters with default values.
- You can only write the function specified; define sub-functions within it if necessary.
- Reuse existing functions as much as possible; this function is one part of the system.
- Avoid global variables; do not use variable names that conflict with API names.
- Preserve the function's docstring (modify its content if needed); keep the function name unchanged.
- Do not use while loops in the function body.
- Do not raise errors or use assertions.
- Do not assume any part of the code; it will be executed directly without modification.
- Import required modules before the function definition, not inside the function body.
- Strictly follow the specified format.
""".strip()

WRITE_LOCAL_FUNCTION_PROMPT_TEMPLATE = WRITE_GLOBAL_FUNCTION_PROMPT_TEMPLATE
