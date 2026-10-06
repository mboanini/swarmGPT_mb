from dataclasses import dataclass
from enum import Enum


class TimingMode(Enum):
    FIXED = "fixed"
    STEPS = "steps"
    TARGET_DURATION = "target_duration"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: str
    description: str
    min_value: int | float | None = None
    max_value: int | float | None = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PrimitiveContract:
    timing_mode: TimingMode
    parameters: tuple[ParameterSpec, ...]

    @property
    def n_args(self) -> int:
        return len(self.parameters)

    def parameter_index(self, name: str) -> int:
        for index, parameter in enumerate(self.parameters):
            if parameter.name == name:
                return index

        raise ValueError(
            f"Parameter '{name}' not found in primitive contract."
        )

    def parameter(self, name: str) -> ParameterSpec:
        return self.parameters[self.parameter_index(name)]