from enum import Enum


class State(Enum):
    NOT_STARTED = 0
    DESIGNED    = 1
    IMPLEMENTED = 2


class FunctionNode:
    def __init__(self, name: str, description: str):
        self.name         = name
        self._description = description
        self._definition  = ""   # Design output: signature + docstring + pass
        self._body        = ""   # Write output:  full implementation
        self._n_args      = 0    # extracted from # n_args: N in definition
        self._state       = State.NOT_STARTED

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value

    @property
    def definition(self):
        return self._definition

    @definition.setter
    def definition(self, value: str):
        self._definition = value

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value: str):
        self._body = value

    @property
    def n_args(self):
        return self._n_args

    @n_args.setter
    def n_args(self, value: int):
        self._n_args = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if isinstance(value, int) and value in range(len(State)):
            self._state = State(value)
        elif isinstance(value, State):
            self._state = value
        else:
            raise ValueError(
                "Invalid state. Must be an integer or an instance of State."
            )

    def reset(self):
        self._definition = ""
        self._body       = ""
        self._n_args     = 0
        self._state      = State.NOT_STARTED
