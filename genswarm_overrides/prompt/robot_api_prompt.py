"""
API disponibili per i droni Crazyflie nel contesto SwarmGPT.
Combina le motion primitives SwarmGPT con le API GenSwarm adattate in 3D.
Coordinate in CENTIMETRI (interi). Bounds: X,Y in [-200,200] cm | Z in [20,200] cm.
Drone IDs numerati da 1.
"""

from modules.framework.parser import CodeParser

robot_api_prompt = """
def get_self_id():
    '''
    Description: Get the unique ID of this drone. IDs start from 1.
    Returns:
    - int: The unique ID of this drone.
    '''

def get_all_drones_id():
    '''
    Description: Get the IDs of all drones in the swarm.
    Returns:
    - list[int]: List of all drone IDs.
    '''

def get_self_position():
    '''
    Description: Get the current 3D position of this drone in real-time.
    Returns:
    - numpy.ndarray: [x, y, z] in centimetres.
    '''

def get_self_velocity():
    '''
    Description: Get the current 3D velocity of this drone.
    Returns:
    - numpy.ndarray: [vx, vy, vz] in cm/s.
    '''

def set_self_velocity(velocity):
    '''
    Description: Set the 3D velocity of this drone immediately.
    Input:
    - velocity (numpy.ndarray): [vx, vy, vz] in cm/s.
    '''

def stop_self():
    '''
    Description: Stop this drone immediately.
    '''

def get_self_radius():
    '''
    Description: Get the safety radius of this drone.
    Returns:
    - float: Radius in centimetres (typically 20 cm).
    '''

def get_environment_range():
    '''
    Description: Get the 3D bounds of the flying volume.
    Returns:
    - dict: {x_min, x_max, y_min, y_max, z_min, z_max} in cm.
    '''

def get_surrounding_environment_info():
    '''
    Description: Get real-time info of surrounding drones and obstacles.
    Returns:
    - list[dict]: each with keys: type, id, position ([x,y,z] cm), velocity ([vx,vy,vz] cm/s), radius.
    '''

def get_all_drones_initial_position():
    '''
    Description: Get the initial 3D positions of all drones.
    Returns:
    - dict[int, numpy.ndarray]: drone_id -> [x, y, z] in cm.
    '''

def get_target_formation_points():
    '''
    Description: Get the 3D target points for the current formation.
    Returns:
    - list[numpy.ndarray]: one [x, y, z] per drone in cm.
    '''

def get_target_position():
    '''
    Description: Get the 3D target position assigned to this drone.
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    '''

def get_prey_position():
    '''
    Description: Get the real-time 3D position of a moving target.
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    '''

def get_lead_position():
    '''
    Description: Get the real-time 3D position of the lead drone.
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    '''

def get_surrounding_unexplored_area():
    '''
    Description: Get unexplored 3D areas within perception range.
    Returns:
    - list[dict]: each with keys: id (int), position ([x,y,z] cm).
    '''

def get_initial_unexplored_areas():
    '''
    Description: Get all initially unexplored areas.
    Returns:
    - list[numpy.ndarray]: each [x, y, z] in cm.
    '''

def get_quadrant_target_position():
    '''
    Description: Get the 3D target for this drone in its assigned quadrant.
    Returns:
    - dict[int, numpy.ndarray]: quadrant_index -> [x, y, z] in cm.
    '''

def get_prey_initial_position():
    '''
    Description: Get the initial 3D position of the prey.
    Returns:
    - numpy.ndarray: [x, y, z] in cm.
    '''

def move(x, y, z, drone_id):
    '''
    Description: Move a drone to an ABSOLUTE 3D position.
    Input:
    - x, y, z (int): Target in cm. X,Y in [-200,200], Z in [20,200]. TARGET, not delta.
    - drone_id (int): The drone to move.
    '''

def move_z(drone_ids, distance):
    '''
    Description: Move drones up or down by a relative distance.
    Input:
    - drone_ids (list[int]): Use [...] for all drones.
    - distance (int): Relative cm (positive=up, negative=down).
    '''

def rotate(angle, axis):
    '''
    Description: Rotate the entire swarm formation.
    Input:
    - angle (float): Degrees.
    - axis (str): "x", "y", or "z".
    '''

def form_circle(drone_ids, z_coord):
    '''
    Description: Arrange drones in a horizontal circle. Radius auto-computed for >= 80cm spacing.
    Input:
    - drone_ids (list[int]): Use [...] for all.
    - z_coord (int): Height in cm, in [20, 200].
    '''

def center(drone_ids):
    '''
    Description: Regroup drones into a tight cluster at the swarm centroid.
    Input:
    - drone_ids (list[int]): Drones to regroup.
    '''

def swap(drone_id_1, drone_id_2):
    '''
    Description: Swap positions of two drones.
    Input:
    - drone_id_1, drone_id_2 (int): The two drones.
    '''

def form_star(height, min_spacing, delta_radius):
    '''
    Description: Arrange drones in a star pattern.
    Input:
    - height (int): Z in cm.
    - min_spacing (int): Min cm between inner ring drones (>= 40).
    - delta_radius (int): Cm between inner and outer ring (>= 40).
    '''

def form_cone(delta_height, spacing, is_inverted):
    '''
    Description: Arrange drones in a 3D cone.
    Input:
    - delta_height (int): Vertical cm between layers.
    - spacing (int): Horizontal cm between drones per layer.
    - is_inverted (bool): True=opens upward, False=opens downward.
    '''

def polygon(n_sides, height):
    '''
    Description: Arrange drones in a regular polygon. Radius auto-computed for >= 60cm spacing.
    Input:
    - n_sides (int): e.g. 3=triangle, 4=square, 6=hexagon.
    - height (int): Z in cm.
    '''
""".strip()


class RobotApi:
    def __init__(self, content):
        self.content = content
        code_obj = CodeParser()
        code_obj.parse_code(self.content)
        self.apis = code_obj.function_defs

        self.base_apis = [
            "get_self_id",
            "get_all_drones_id",
            "get_self_position",
            "set_self_velocity",
            "get_self_radius",
            "get_surrounding_environment_info",
            "get_all_drones_initial_position",
            "get_environment_range",
        ]

        self.api_scope = {
            "get_self_id":                      ["local"],
            "get_all_drones_id":                ["global"],
            "get_self_position":                ["local"],
            "get_self_velocity":                ["local"],
            "set_self_velocity":                ["local"],
            "stop_self":                        ["local"],
            "get_self_radius":                  ["local"],
            "get_environment_range":            ["local", "global"],
            "get_surrounding_environment_info": ["local"],
            "get_all_drones_initial_position":  ["global"],
            "get_target_formation_points":      ["global"],
            "get_target_position":              ["local"],
            "get_prey_position":                ["local"],
            "get_lead_position":                ["local"],
            "get_prey_initial_position":        ["global"],
            "get_surrounding_unexplored_area":  ["local"],
            "get_initial_unexplored_areas":     ["global"],
            "get_quadrant_target_position":     ["global", "local"],
            "move":         ["global"],
            "move_z":       ["global"],
            "rotate":       ["global"],
            "form_circle":  ["global"],
            "center":       ["global"],
            "swap":         ["global"],
            "form_star":    ["global"],
            "form_cone":    ["global"],
            "polygon":      ["global"],
        }

        self.base_prompt = [self.apis[api] for api in self.base_apis if api in self.apis]

        self.task_apis = {
            "bridging":     ["stop_self"],
            "aggregation":  ["stop_self", "center"],
            "covering":     ["get_environment_range", "stop_self", "form_circle", "polygon"],
            "crossing":     ["stop_self"],
            "encircling":   ["get_prey_position", "get_prey_initial_position", "form_circle"],
            "exploration":  ["get_initial_unexplored_areas", "get_surrounding_unexplored_area",
                             "get_environment_range", "stop_self"],
            "flocking":     ["get_environment_range", "get_self_velocity"],
            "clustering":   ["get_quadrant_target_position"],
            "shaping":      ["get_target_formation_points", "stop_self",
                             "form_circle", "polygon", "form_star", "form_cone"],
            "pursuing":     ["get_lead_position"],
            "formation":    ["form_circle", "polygon", "form_star", "form_cone", "center", "rotate", "move"],
            "free":         list(self.api_scope.keys()),
        }

    def get_api_prompt(self, task_name=None, scope=None, only_names=False):
        if task_name is None:
            all_apis = list(self.apis.keys())
            if scope:
                all_apis = [a for a in all_apis if scope in self.api_scope.get(a, [])]
            if only_names:
                return all_apis
            return "\n\n".join([self.apis[a] for a in all_apis])

        task_prompt = self.base_prompt.copy()
        specific = [self.apis[api] for api in self.task_apis.get(task_name, []) if api in self.apis]
        task_prompt.extend(specific)

        if scope:
            task_prompt = [
                a for a in task_prompt
                if scope in self.api_scope.get(self.get_api_name(a), [])
            ]

        if only_names:
            return [self.get_api_name(a) for a in task_prompt]
        return "\n\n".join(task_prompt)

    def get_api_name(self, api_content):
        return next((name for name, c in self.apis.items() if c == api_content), None)


robot_api = RobotApi(content=robot_api_prompt)

ALLOCATOR_TEMPLATE = """

def get_assigned_task():
    '''
    Description: Get the task assigned to this drone by the global allocator.
    Returns:
        {template}
    '''
"""