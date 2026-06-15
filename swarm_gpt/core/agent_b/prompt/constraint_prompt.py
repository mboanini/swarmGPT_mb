CONSTRAINTS_TEXT = """
    - All positions in cm. `limits` values are in metres — multiply by 100 to compare.
    - Min 60 cm between any two drones at every waypoint.
    - Max 100 cm/s velocity per drone between consecutive waypoints.
    - Waypoint timestamps strictly in (tstart, tend]. At least one waypoint required.
    - final_pos shape: (n_drones, 3).
""".strip()