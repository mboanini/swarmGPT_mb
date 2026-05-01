from swarmgpt_global_apis import get_all_drones_id,get_all_drones_initial_position,get_environment_range,get_all_drones_id,get_environment_range,get_all_drones_initial_position,get_target_formation_points,get_prey_initial_position,get_initial_unexplored_areas,get_quadrant_target_position,move,move_z,rotate,form_circle,center,swap,form_star,form_cone,polygon
import numpy as np

def fetch_environment_range():
    """
    Description: Retrieve the 3D bounds of the flying volume. This function doesn't take inputs and returns the bounds as a dictionary.
    
    Returns:
        dict: A dictionary containing the bounds of the flying volume with keys {x_min, x_max, y_min, y_max, z_min, z_max} and corresponding integer values in centimeters.
    """
    # Call the global API to get the environment range
    environment_bounds = get_environment_range()
    
    return environment_bounds


def fetch_all_drones_info():
    """
    Description:
    Fetch the IDs and initial positions of all drones in the flying volume.
    This function gathers information about all drones using available APIs,
    integrating the drone IDs and their initial 3D positions into a single dictionary.
    
    Params:
        None: This function does not take any parameters.
    
    Return:
        dict: A dictionary where the keys are drone IDs (int) and the values are 
              their respective initial positions as numpy arrays [x, y, z] in cm.
              Example: {1: np.array([0, 0, 50]), 2: np.array([10, -10, 70]), ...}
    """
    # Retrieve the list of all drone IDs
    drone_ids = get_all_drones_id()
    
    # Retrieve the initial positions of all drones
    initial_positions = get_all_drones_initial_position()
    
    # Combine the two into a single dictionary
    drone_info = {drone_id: np.array(position) for drone_id, position in initial_positions.items()}
    
    return drone_info


def assign_quadrants_to_drones(bounds, drone_info):
    """
    Description:
    Divide the flying volume into quadrants and assign each drone a distinct quadrant to explore. 
    Takes the bounds of the flying volume and the initial positions of all drones. Returns a dictionary 
    mapping drone_ids to their respective target quadrants.
    
    Params:
        bounds (dict): A dictionary containing the bounds of the flying volume with keys {x_min, x_max, y_min, y_max, z_min, z_max} and corresponding integer values in centimeters. Example: {'x_min': -200, 'x_max': 200, 'y_min': -200, 'y_max': 200, 'z_min': 20, 'z_max': 200}.
        drone_info (dict): A dictionary where the keys are drone IDs (int) and the values are their respective initial positions as numpy arrays [x, y, z] in cm. Example: {1: np.array([0, 0, 50]), 2: np.array([10, -10, 70]), ...}.
    
    Returns:
        dict: A dictionary mapping drone IDs to their respective quadrants as numpy arrays [x, y, z] representing the center of the assigned quadrant in cm.
    """
    
    # Calculate the midpoints in each dimension to divide the space into 8 quadrants
    mid_x = (bounds['x_min'] + bounds['x_max']) // 2
    mid_y = (bounds['y_min'] + bounds['y_max']) // 2
    mid_z = (bounds['z_min'] + bounds['z_max']) // 2
    
    # Generate the centers of the 8 quadrants
    quadrants_centers = [
        np.array([(x1 + x2) // 2, (y1 + y2) // 2, (z1 + z2) // 2])
        for x1, x2 in [(bounds['x_min'], mid_x), (mid_x, bounds['x_max'])] 
        for y1, y2 in [(bounds['y_min'], mid_y), (mid_y, bounds['y_max'])]
        for z1, z2 in [(bounds['z_min'], mid_z), (mid_z, bounds['z_max'])]
    ]
    
    # Ensure there are at least as many quadrants as drones
    num_drones = len(drone_info)
    if num_drones > len(quadrants_centers):
        raise Exception("Not enough quadrants for the number of drones.")
    
    # Assign drones to nearest quadrants
    drone_to_quadrant = {}
    unassigned_quadrants = quadrants_centers.copy()
    
    for drone_id, initial_pos in drone_info.items():
        min_distance = float('inf')
        closest_quadrant = None
        for quadrant in unassigned_quadrants:
            distance = np.linalg.norm(initial_pos - quadrant)
            if distance < min_distance:
                min_distance = distance
                closest_quadrant = quadrant
        drone_to_quadrant[drone_id] = closest_quadrant
        unassigned_quadrants.remove(closest_quadrant)  # Remove the assigned quadrant from the list
    
    return drone_to_quadrant
