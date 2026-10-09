from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class RuntimeScenario:
    swarm_pos: np.ndarray
    tstart: float
    tend: float
    limits: dict[str, np.ndarray]

    