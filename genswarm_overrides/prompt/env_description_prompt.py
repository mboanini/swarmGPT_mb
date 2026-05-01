ENV_DES = """
Environment:
    3D bounded volume. Coordinates in centimeters (integers only).
    Bounds: X, Y in [-200, 200] cm | Z in [20, 200] cm
    Multiple Crazyflie 2.1 nano-quadrotors share the airspace.

Drone (Crazyflie 2.1):
    Max speed: configurable cm/s
    Min distance between any two drones at all times: 40 cm
    Drone IDs are numbered starting from 1.
    A safety filter (AMSwarm) post-processes all trajectories to enforce
    collision avoidance and kinematic feasibility.
""".strip()