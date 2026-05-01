from scipy.optimize import linear_sum_assignment
from swarmgpt_global_apis import get_all_drones_id,get_all_drones_initial_position,get_environment_range,get_all_drones_id,get_environment_range,get_all_drones_initial_position,get_target_formation_points,get_prey_initial_position,get_initial_unexplored_areas,get_quadrant_target_position,move,move_z,rotate,form_circle,center,swap,form_star,form_cone,polygon
import numpy as np

def assign_goals_to_drones():
    """
    Description: This function allocates sub-goals or target positions to each drone to achieve the overall flocking behavior while avoiding obstacles and maintaining required distance constraints.
    
    params:
        None: This function retrieves necessary data using existing APIs internally.
    return:
        goals (dict): A dictionary where the key is the drone ID and the value is the assigned [x, y, z] target position for that drone.
    """
    # Fetch initial data
    drone_ids = get_all_drones_id()
    initial_positions = get_all_drones_initial_position()
    target_positions = get_target_formation_points()
    
    num_drones = len(drone_ids)
    
    # Convert positions to a manageable format
    initial_pos_mat = np.array([initial_positions[drone_id] for drone_id in drone_ids])
    target_pos_mat = np.array(target_positions)
    
    # Calculate the cost matrix (distance between each initial position and each target position)
    cost_matrix = np.linalg.norm(
        initial_pos_mat[:, np.newaxis] - target_pos_mat[np.newaxis, :], axis=2)
    
    # Apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Assign the goals based on the optimal assignment
    goals = {drone_ids[row]: target_pos_mat[col].tolist() for row, col in zip(row_ind, col_ind)}
    
    return goals
