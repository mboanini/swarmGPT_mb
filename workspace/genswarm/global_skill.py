from swarmgpt_global_apis import get_all_drones_id,get_all_drones_initial_position,get_environment_range,get_all_drones_id,get_environment_range,get_all_drones_initial_position,get_target_formation_points,get_prey_initial_position,get_initial_unexplored_areas,get_quadrant_target_position,move,move_z,rotate,form_circle,center,swap,form_star,form_cone,polygon
import numpy as np
import time

def form_square(drone_ids=None, z_height=100):
    """
    Description:
        Arrange drones in a square formation with equal spacing between all adjacent drones.
        The function:
        - Fetches the drone IDs and initial positions.
        - Computes optimal square formation based on the number of drones and environment bounds.
        - Moves the drones to the computed positions.
    
    params:
        drone_ids: list, Optional list of drone IDs. If not provided, all available drones will be used.
        z_height: int, Optional height in cm at which the square formation should be created. Defaults to 100 cm.
    
    return:
        None
    """
    
    if drone_ids is None:
        drone_ids = get_all_drones_id()
    
    num_drones = len(drone_ids)
    
    if num_drones < 4:
        return  # Not enough drones to form a square
    
    side_length = int(np.ceil(np.sqrt(num_drones)))
    spacing = 40  # Minimum spacing requirement

    # Compute initial positions of the grid
    formation_positions = []
    for i in range(int(side_length)):
        for j in range(int(side_length)):
            idx = i * int(side_length) + j
            if idx < num_drones:
                x = j * spacing
                y = i * spacing
                formation_positions.append((x, y, z_height))

    # Center the grid in the environment bounds
    x_min, x_max, y_min, y_max, z_min, z_max = get_environment_range().values()
    x_offset = (x_max + x_min) / 2 - (side_length - 1) * spacing / 2
    y_offset = (y_max + y_min) / 2 - (side_length - 1) * spacing / 2
    
    # Move drones to their target positions
    for drone_id, pos in zip(drone_ids, formation_positions):
        x, y, z = pos
        move(x + x_offset, y + y_offset, z, drone_id)
    
    # Move 1 meter to the right
    time.sleep(1)  # Assuming some delay before the next movement
    for drone_id, pos in zip(drone_ids, formation_positions):
        x, y, z = pos
        move(x + x_offset + 100, y + y_offset, z, drone_id)


def monitor_and_maintain_square(monitor_interval=0.5):
    """
    Description:
    Continuously monitor and adjust drone positions to maintain the square formation,
    and move the entire formation 1 meter (100 cm) to the right (positive X direction).
    The function ensures that drones maintain the desired relative positions at all times,
    taking into account potential environmental disturbances. The function runs a loop
    to repeatedly check and correct drone positions until they reach the target location.
    
    params:
        monitor_interval: float, The time interval (in seconds) between successive monitoring checks.
        
    return:
        None
    """
    
    def get_positions(drone_ids):
        positions = get_all_drones_initial_position()
        return {drone_id: positions[drone_id] for drone_id in drone_ids}
    
    def monitor_and_correct_formation(drone_ids, target_positions):
        current_positions = get_positions(drone_ids)
        for drone_id in drone_ids:
            current_pos = current_positions[drone_id]
            target_pos = target_positions[drone_id]
            # Check if any drone is significantly deviating
            if np.linalg.norm(current_pos - target_pos) > 5:  # example threshold of 5 cm
                move(target_pos[0], target_pos[1], target_pos[2], drone_id)
        
    # Step 1: Fetch required data
    drone_ids = get_all_drones_id()
    
    # Step 2: Form initial square
    form_square(drone_ids)
    
    # Step 3: Move the square formation to the right by 100 cm as a whole
    move_square_formation()
    
    # Step 4: Monitor and maintain the formation
    initial_positions = get_positions(drone_ids)
    target_positions = {drone_id: pos + np.array([100, 0, 0]) for drone_id, pos in initial_positions.items()}
    
    for _ in range(20):  # Adjust the range value as needed to ensure stability
        monitor_and_correct_formation(drone_ids, target_positions)
        time.sleep(monitor_interval)


def move_square_formation():
    """
    Description: Move the entire square formation of drones 1 meter (100 cm) to the right (positive X direction) while maintaining the relative positions of the drones.
    
    params:
        None
    return:
        None
    """
    # Step 1: Get initial positions of all drones
    drone_positions = get_all_drones_initial_position()
    
    # Step 2: Compute target positions
    target_positions = {drone_id: pos + np.array([100, 0, 0]) for drone_id, pos in drone_positions.items()}
    
    # Step 3: Move each drone to its new target position
    for drone_id, target_pos in target_positions.items():
        move(target_pos[0], target_pos[1], target_pos[2], drone_id)
