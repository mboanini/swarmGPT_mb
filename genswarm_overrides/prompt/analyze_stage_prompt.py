ANALYZE_SKILL_PROMPT_TEMPLATE: str = """
## Background:
{task_des}
## Role setting:
- You are a function designer. You need to design functions based on user commands and constraint information.

## These are the environment description:
These are the basic descriptions of the environment.
{env_des}

## These are the User original instructions:
{instruction}

## These are the APIs available to the orchestrator:
All APIs run on the centralized orchestrator. There is no per-drone local code.

### Query APIs (read global state):
```python
{local_api}
```

### Execution APIs (motion primitives and global queries):
```python
{global_api}
```

## Constraints information:
The following are the constraints that the generated functions need to satisfy.
{constraints}


## The output TEXT format is as follows:
{output_template}

## Notes:
- Analyze the essential functions needed to implement the user commands.
- All functions run on the centralized orchestrator. There is no distinction between
  "local" (per-drone) and "global" (allocator) code — everything is orchestrator code.
- Each function should either EXECUTE behavior (call motion/execution APIs to move drones)
  or compute a sub-result used by another function that does the execution.
- Functions that execute behavior should return None.
- Functions that compute sub-results (e.g., target positions) should return that result,
  which is then consumed by a caller that invokes the motion APIs.
- Each function should be decoupled from others but able to cooperate and call each other.
- Each function should implement a small part of the overall objective.
- Each function must satisfy the relevant constraints listed.
- One function can satisfy multiple constraints; multiple functions can address one constraint.
- Only the names of the functions, their constraints, and their call relationships are required.
- The inter-call relationships among these functions must be determined.
- There should be no functional redundancy; each function has a distinct responsibility.
- Each constraint must be fulfilled by one of the functions listed.
- The output should strictly adhere to the specified format.
""".strip()

ANALYZE_CONSTRAINT_PROMPT_TEMPLATE: str = """
## Background:
{task_des}
## Role setting:
- You need to analyze what functional constraints are needed to meet the user's requirements.

## These are the environment description:
These are the basic descriptions of the environment.
{env_des}

## These are the APIs available to the orchestrator:
All APIs run on the centralized orchestrator. There is no per-drone local code.

### Query APIs (read global state):
```python
{local_api}
```

### Execution APIs (motion primitives and global queries):
```python
{global_api}
```

## User commands:
{instruction}


## The output TEXT format is as follows:
{output_template}

## Notes:
Your output should satisfy the following notes:
- Constraints should not be too simple or too complex; the amount of code required to
  implement each constraint should be similar.
- Constraints should be practical and achievable through writing orchestrator code.
- Constraints describe what the orchestrator must achieve in terms of swarm behavior.
- Each constraint will correspond to at least one executable function, and the combination
  of all constraints can meet the user's needs.
- Analyze the core tasks proposed by the user and decompose them into functional constraints.
- You need to understand the existing APIs. The capabilities provided by these APIs have
  already been implemented; the orchestrator can call them directly.
- There's no need to regenerate constraints already covered by the APIs.
- These constraints should be significant and mutually independent.
- If the user's instruction involves specific numerical values, retain them in the constraint description.
- The output should strictly adhere to the specified format.
""".strip()

CONSTRAIN_TEMPLATE: str = """
##reasoning: (you should think step by step, and analyze the constraints that need to be satisfied in the task.place the analysis results at here.)
```json
{
  "constraints": [
    {
      "name": "Constraint name",
      "description": "Description of the constraint.(If the user's requirements involve specific numerical values, they should be reflected in the description. )"
    },
  ]
}
```
""".strip()

FUNCTION_TEMPLATE: str = """
##Reasoning: (Think step by step, and analyze the functions that need to be implemented in the task. All functions run on the centralized orchestrator.)
```json
{
  "functions": [
    {
      "name": "Function name",
      "description": "Description of the function, including input/output parameters. If the function executes movement, state that it returns None and calls motion APIs.",
      "constraints": [
        "Name of the constraint that this function needs to satisfy"
      ],
      "calls": [
        "Function name or API that this function calls"
      ],
      "scope": "global"
    }
  ]
}
```
""".strip()
