TASK_DES = """
You are writing a new motion primitive for SwarmGPT, a Crazyflie 2.1 drone swarm controller. 
Motion primitives are Python functions that receive the current swarm state and a time window, 
and return target positions as a waypoint dictionary.
A single mistake in units, limits or array mutation can cause drone collisions. 
Follow every constraint exactly - they are safety requirements, not style guidelines.
""".strip()