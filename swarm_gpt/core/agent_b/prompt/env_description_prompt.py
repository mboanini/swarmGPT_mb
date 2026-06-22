ENV_DES = """
Indoor arena with rectangular spatial limits.
    - Coordinates are in centimetres (cm).
    - Bounds: X, Y in [-200, 200] cm | Z in [20, 200] cm.
    - limits["lower"] and limits["upper"] are in METERS — multiply by 100 for cm.
    - `swarm_pos` shape: (n_drones, 3) in cm. n_drones = swarm_pos.shape[0].
    - Max velocity between consecutive waypoints: 100 cm/s.
    - Min distance between any two drones at every waypoint: 60 cm.
""".strip()