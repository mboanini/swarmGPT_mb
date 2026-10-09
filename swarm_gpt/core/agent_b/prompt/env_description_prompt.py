ENV_DES = """
Indoor arena with rectangular spatial limits.
    - Coordinates are in centimetres (cm).
    - The drones are limited to [-200, 200] in x, [-200, 200] in y, [0, 200] in z
    - The drones must not touch the ground, i.e. their z coordinate must always be greater than 0 
    - limits["lower"] and limits["upper"] are in METERS — multiply by 100 for cm.
    - `swarm_pos` shape: (n_drones, 3) in cm. n_drones = swarm_pos.shape[0].
    - Max velocity between consecutive waypoints: 100 cm/s.
    - Min distance between any two drones at every waypoint: 40 cm.
""".strip()