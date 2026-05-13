"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

import numpy as np
from itertools import combinations

from scipy.spatial import Delaunay, procrustes, distance_matrix

from run.utils import calculate_overlap_ratio
import numpy as np
from scipy.optimize import leastsq


def check_collisions(data, tolerance: float = 0.1) -> dict:
    """
    Check if there are any collisions between robots and obstacles at any time step.
    :param data: dictionary containing information about entities
    :param tolerance: allowed tolerance for collision
    :return:
        a dictionary containing:
        - collision (bool): Whether any collisions occurred.
        - collision_count (int): The total number of collisions across all time steps.
        - collision_severity_sum (float): The sum of overlap ratios for all collisions.
    """
    robots = {id: info for id, info in data.items() if info["type"] == "Robot"}
    obstacles = {id: info for id, info in data.items() if info["type"] == "Obstacle"}
    num_timesteps = len(next(iter(data.values()))["trajectory"])
    collision_count = 0
    collision_severity_sum = 0

    for t in range(num_timesteps):
        # Check for robot-obstacle collisions
        for robot_id, robot_info in robots.items():
            robot_position = robot_info["trajectory"][t]
            for _, obstacle_info in obstacles.items():
                overlap_ratio = calculate_overlap_ratio(
                    robot_position,
                    obstacle_info["trajectory"][0],
                    robot_info["size"],
                    obstacle_info["size"],
                    tolerance,
                )
                if overlap_ratio is not None:
                    collision_count += 1
                    collision_severity_sum += overlap_ratio

        # Check for robot-robot collisions
        for (id1, robot1_info), (id2, robot2_info) in combinations(robots.items(), 2):
            robot1_position = robot1_info["trajectory"][t]
            robot2_position = robot2_info["trajectory"][t]
            overlap_ratio = calculate_overlap_ratio(
                robot1_position,
                robot2_position,
                robot1_info["size"],
                robot2_info["size"],
                tolerance,
            )
            if overlap_ratio is not None:
                collision_count += 1
                collision_severity_sum += overlap_ratio
    collision = collision_count > 0
    return {
        "collision": collision,
        "collision_count": collision_count,
        "collision_severity_sum": collision_severity_sum,
    }


def evaluate_target_achievement(data, tolerance=0.1) -> dict:
    """
    Evaluate the performance of entities in achieving their targets.

    :param data: dictionary containing information about entities, including their trajectories and targets.
    :param tolerance: the distance threshold within which a target is considered achieved.
    :return: A dict containing:
        - all_targets_achieved (bool): Whether all targets were achieved by all entities.
        - target_achievement_ratio (float): The ratio of entities that achieved their targets.
        - average_distance_ratio (float): The average ratio of final distance to initial distance for all entities.
        - average_steps_ratio (float or str): The average ratio of steps taken to total steps available for entities that achieved their targets. Returns 'inf' if no targets were achieved.
    """
    total_distance_ratio = 0
    achieved_targets = 0
    num_targets = 0
    total_steps_ratio = 0

    for entity_id, info in data.items():
        target = info.get("target")
        if target is None:
            continue

        num_targets += 1
        initial_position = info["trajectory"][0]
        initial_distance = np.linalg.norm(target - initial_position)
        steps_to_target = len(info["trajectory"])
        final_distance = initial_distance

        for t, position in enumerate(info["trajectory"]):
            current_distance = np.linalg.norm(target - position)
            if current_distance <= tolerance:
                steps_to_target = t + 1
                final_distance = current_distance
                break

        if np.linalg.norm(target - info["trajectory"][-1]) <= tolerance:
            achieved_targets += 1
            total_steps_ratio += steps_to_target / len(info["trajectory"])
        else:
            print(np.linalg.norm(target - info["trajectory"][-1]))

        distance_ratio = (
            final_distance / initial_distance if initial_distance > 0 else 1
        )
        total_distance_ratio += distance_ratio

        print(
            f"Entity {entity_id}: Initial Distance: {initial_distance}, Final Distance: {final_distance}, "
            f"Ratio: {distance_ratio}, Steps to Target: {steps_to_target}"
        )

    all_targets_achieved = achieved_targets == num_targets
    target_achievement_ratio = achieved_targets / num_targets if num_targets > 0 else 0
    average_distance_ratio = (
        total_distance_ratio / num_targets if num_targets > 0 else 0
    )

    return {
        "all_targets_achieved": all_targets_achieved,
        "target_achievement_ratio": target_achievement_ratio,
        "average_distance_ratio": average_distance_ratio,
    }


def evaluate_landmark_visits(data) -> dict:
    """
    Evaluate the visitation of landmarks by entities.

    :param data: dictionary containing information about entities, including their types and states.
    :return: A dictionary containing:
        - 'all_landmarks_visited' (bool): Whether all landmarks were visited.
        - 'landmark_visit_ratio' (float): The ratio of landmarks that were visited.
        - 'average_visit_step' (float): The average time step at which landmarks were visited.
    """
    visited_landmarks = 0
    num_landmarks = 0
    visit_steps = []

    for entity_id, info in data.items():
        if info.get("type") == "Landmark":
            num_landmarks += 1
            if "visited" in info.get("states", []):
                visited_landmarks += 1
                visit_step = info["states"].index("visited")
                print(f"Landmark {entity_id} visited at time step {visit_step}")
                visit_steps.append(visit_step)

    all_landmarks_visited = visited_landmarks == num_landmarks
    landmark_visit_ratio = visited_landmarks / num_landmarks if num_landmarks > 0 else 0
    average_visit_step = np.mean(visit_steps) if visit_steps else 0

    return {
        "all_landmarks_visited": all_landmarks_visited,
        "landmark_visit_ratio": landmark_visit_ratio,
        "average_visit_step": average_visit_step,
    }


def calculate_line_similarity(data, target_line: tuple) -> dict:
    """
    Calculate the similarity between the line formed by the robots' final positions and a given target line.

    :param data: dictionary containing information about entities, including their trajectories.
    :param target_line: a tuple containing two points (x1, y1) and (x2, y2) defining the target line.
    :return: A dictionary containing:
        - length_similarity (float): The ratio of the lengths between the fitted line and the target line.
        - slope_similarity (float): The cosine similarity between the slopes of the fitted line and the target line.
        - point_similarity (float): The average Euclidean distance between the points on the fitted line and the target line.
    """
    robot_positions = []
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions.append(
                info["trajectory"][-1]
            )  # Get final positions of robots

    if len(robot_positions) < 2:
        raise ValueError("Not enough robots to form a line.")

    # Fit a line to the robots' final positions
    robot_positions = np.array(robot_positions)
    A = np.vstack([robot_positions[:, 0], np.ones(len(robot_positions))]).T
    slope, intercept = np.linalg.lstsq(A, robot_positions[:, 1], rcond=None)[0]

    # Fitted line: y = slope * x + intercept
    fitted_line = ((0, intercept), (1, slope + intercept))

    # Calculate length similarity
    target_length = np.linalg.norm(np.array(target_line[1]) - np.array(target_line[0]))
    fitted_length = np.linalg.norm(np.array(fitted_line[1]) - np.array(fitted_line[0]))
    length_similarity = (
        min(fitted_length / target_length, target_length / fitted_length)
        if target_length > 0
        else 0
    )

    # Calculate slope similarity (using cosine similarity)
    target_slope = (
        (target_line[1][1] - target_line[0][1])
        / (target_line[1][0] - target_line[0][0])
        if target_line[1][0] != target_line[0][0]
        else np.inf
    )
    slope_similarity = np.dot([1, slope], [1, target_slope]) / (
            np.linalg.norm([1, slope]) * np.linalg.norm([1, target_slope])
    )

    # Calculate point similarity (average distance between points on the fitted line and target line)
    point_distances = []
    for point in robot_positions:
        target_y = target_slope * point[0] + (
                target_line[0][1] - target_slope * target_line[0][0]
        )
        fitted_y = slope * point[0] + intercept
        point_distances.append(np.abs(target_y - fitted_y))
    point_similarity = np.mean(point_distances)

    return {
        "length_similarity": length_similarity,
        "slope_similarity": slope_similarity,
        "point_similarity": point_similarity,
    }


def evaluate_robot_final_positions(data) -> dict:
    """
    Evaluate various metrics related to the final positions of robots.

    :param data: dictionary containing information about entities, including their trajectories.
    :return: A dictionary containing:
        - mean_nearest_neighbor_distance (float): The mean distance between each robot and its nearest neighbor.
        - variance_nearest_neighbor_distance (float): The variance of the distances between each robot and its nearest neighbor.
        - mean_displacement (float): The mean displacement of robots from their initial positions to their final positions.
        - area_ratio (float): The ratio of the area occupied by the robots' final positions to 25.
    """
    robot_positions_initial = []
    robot_positions_final = []

    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions_initial.append(info["trajectory"][0])
            robot_positions_final.append(info["trajectory"][-1])

    robot_positions_initial = np.array(robot_positions_initial)
    robot_positions_final = np.array(robot_positions_final)

    # Calculate nearest neighbor distances for final positions
    nearest_neighbor_distances = []
    for i, pos1 in enumerate(robot_positions_final):
        distances = np.linalg.norm(robot_positions_final - pos1, axis=1)
        distances[i] = np.inf  # Exclude the distance to itself
        nearest_neighbor_distances.append(np.min(distances))

    mean_nearest_neighbor_distance = np.mean(nearest_neighbor_distances)
    variance_nearest_neighbor_distance = np.var(nearest_neighbor_distances)

    # Calculate mean displacement from initial to final positions
    displacements = np.linalg.norm(
        robot_positions_final - robot_positions_initial, axis=1
    )
    mean_displacement = np.mean(displacements)

    # Calculate the area of the bounding box formed by the robots' final positions
    x_min, y_min = np.min(robot_positions_final, axis=0)
    x_max, y_max = np.max(robot_positions_final, axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    print("(x_max - x_min), (y_max - y_min)", (x_max - x_min), (y_max - y_min))
    area_ratio = area / 16

    return {
        "mean_nearest_neighbor_distance": mean_nearest_neighbor_distance,
        "variance_nearest_neighbor_distance": variance_nearest_neighbor_distance,
        "mean_displacement": mean_displacement,
        "area_ratio": area_ratio,
    }


def fit_circle(positions):
    """
    Fit a circle to the given 2D positions using least squares.

    :param positions: A numpy array of shape (N, 2) representing N robot positions (x, y).
    :return: A tuple containing (fitted_center_x, fitted_center_y, fitted_radius).
    """

    # Function to calculate the algebraic distance between the data points and the circle
    def calc_R(xc, yc):
        return np.sqrt((positions[:, 0] - xc) ** 2 + (positions[:, 1] - yc) ** 2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Initial guess for circle center
    center_estimate = np.mean(positions, axis=0)
    center_final, _ = leastsq(f, center_estimate)
    Ri_final = calc_R(*center_final)
    radius_final = Ri_final.mean()

    return center_final[0], center_final[1], radius_final


def evaluate_robot_circle_similarity(
        data, expected_circle_center: tuple, expected_circle_radius: float
) -> dict:
    """
    Evaluate the similarity between the robots' final positions and a fitted circle,
    and compare the fitted circle with the specified expected circle.

    :param data: dictionary containing information about entities, including their trajectories.
    :param expected_circle_center: a tuple representing the (x, y) coordinates of the expected circle's center.
    :param expected_circle_radius: the radius of the expected circle.
    :return: A dictionary containing:
        - fitted_circle_center (tuple): The fitted circle's center (x, y).
        - fitted_circle_radius (float): The radius of the fitted circle.
        - center_distance (float): Distance between the fitted and expected circle centers.
        - radius_difference (float): The absolute difference between the fitted and expected radii.
        - angle_standard_deviation (float): The standard deviation of the angle differences between adjacent robots.
        - angle_max_min_difference (float): The difference between the maximum and minimum angle intervals.
    """
    robot_positions_final = []

    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions_final.append(info["trajectory"][-1])

    robot_positions_final = np.array(robot_positions_final)
    num_robots = len(robot_positions_final)

    if num_robots == 0:
        raise ValueError("No robots found in the data.")

    # Step 1: Fit a circle to the robot positions
    fitted_circle_center_x, fitted_circle_center_y, fitted_circle_radius = fit_circle(
        robot_positions_final
    )
    fitted_circle_center = (fitted_circle_center_x, fitted_circle_center_y)

    # Step 2: Calculate the similarity to the expected circle
    center_distance = np.linalg.norm(
        np.array(fitted_circle_center) - np.array(expected_circle_center)
    )
    radius_difference = np.abs(fitted_circle_radius - expected_circle_radius)

    # Step 3: Calculate the angle of each robot with respect to the fitted circle center
    angles = []
    for position in robot_positions_final:
        dx, dy = position - np.array(fitted_circle_center)
        angle = np.arctan2(dy, dx)
        angles.append(angle)

    # Normalize angles to be within [0, 2*pi]
    angles = np.array(angles)
    angles = np.mod(angles, 2 * np.pi)

    # Sort the angles
    angles.sort()

    # Calculate angle intervals between adjacent robots
    angle_intervals = np.diff(angles, append=angles[0] + 2 * np.pi)

    # Step 4: Calculate angle standard deviation and max-min difference
    angle_standard_deviation = np.std(angle_intervals)
    angle_max_min_difference = np.max(angle_intervals) - np.min(angle_intervals)

    return {
        "center_distance": center_distance,
        "radius_difference": radius_difference,
        "angle_standard_deviation": angle_standard_deviation,
        "angle_max_min_difference": angle_max_min_difference,
    }


def evaluate_trajectory_pattern(data) -> dict:
    """
    Evaluate the trajectory pattern by analyzing the spatial distribution and overall shape of robot final positions.

    :param data: dictionary containing information about entities, including their trajectories.
    :return: A dictionary containing:
        - spatial_variance (float): Variance of the robot positions indicating the spread.
        - bounding_box_area (float): Area of the bounding box enclosing all final positions of the robots.
    """
    robot_positions_final = []

    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions_final.append(
                info["trajectory"][-1]
            )  # Use the final position of each robot

    robot_positions_final = np.array(robot_positions_final)
    num_robots = len(robot_positions_final)

    if num_robots == 0:
        raise ValueError("No robots found in the data.")

    # Calculate the spatial variance of robot positions
    spatial_variance = np.var(robot_positions_final, axis=0).sum()

    # Calculate the area of the bounding box formed by the robots' final positions
    x_min, y_min = np.min(robot_positions_final, axis=0)
    x_max, y_max = np.max(robot_positions_final, axis=0)
    bounding_box_area = (x_max - x_min) * (y_max - y_min)

    return {
        "spatial_variance": spatial_variance,
        "bounding_box_area": bounding_box_area,
    }


def evaluate_trajectory_similarity(data) -> dict:
    """
    Evaluate the similarity between the trajectories of all robots using Dynamic Time Warping (DTW).

    :param data: dictionary containing information about entities, including their trajectories.
    :return: A dictionary containing:
        - mean_dtw_distance (float): The mean DTW distance between all pairs of robot trajectories.
        - variance_dtw_distance (float): The variance of DTW distances between all pairs of robot trajectories.
    """
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    trajectories = []

    for entity_id, info in data.items():
        if info["type"] == "Robot":
            trajectories.append(np.array(info["trajectory"]))
        if info["type"] == "Prey":
            trajectories.append(np.array(info["trajectory"]))

    num_robots = len(trajectories)

    if num_robots < 2:
        raise ValueError("Not enough robots to calculate trajectory similarity.")

    dtw_distances = []

    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            distance, _ = fastdtw(trajectories[i], trajectories[j], dist=euclidean)
            dtw_distances.append(distance)

    mean_dtw_distance = np.mean(dtw_distances)
    variance_dtw_distance = np.var(dtw_distances)

    return {
        "mean_dtw_distance": mean_dtw_distance,
        "variance_dtw_distance": variance_dtw_distance,
    }


def evaluate_min_distances_to_others(data) -> dict:
    """
    Evaluate the minimum distance between each robot's final position and all other robots' final positions.

    :param data: dictionary containing information about entities, including their final positions.
    :return: A dictionary containing:
        - max_min_distance (float): The maximum of the minimum distances between each robot's final position and the others.
    """
    from scipy.spatial.distance import euclidean
    robot_radius = 0.10
    final_positions = []
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            # Assuming the final position is the last point in the trajectory
            final_positions.append(np.array(info["trajectory"][-1]))
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_radius = info["size"]

    num_robots = len(final_positions)

    if num_robots < 2:
        raise ValueError("Not enough robots to calculate distances.")

    min_distances = []

    for i in range(num_robots):
        distances_to_others = []
        for j in range(num_robots):
            if i != j:
                distance = euclidean(final_positions[i], final_positions[j])
                distance = distance - 2 * robot_radius
                distances_to_others.append(distance)
        min_distances.append(min(distances_to_others))

    max_min_distance = max(min_distances)

    return {
        "max_min_distance": max_min_distance,
    }


def evaluate_average_position(data) -> dict:
    """
    Evaluate the average position of all robots.

    :param data: dictionary containing information about entities, including their final positions.
    :return: A dictionary containing:
        - average_position (tuple): The average position of all robots.
    """
    robot_positions_final = []

    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions_final.append(info["trajectory"][-1])

    robot_positions_final = np.array(robot_positions_final)
    num_robots = len(robot_positions_final)

    if num_robots == 0:
        raise ValueError("No robots found in the data.")

    average_position = np.mean(robot_positions_final, axis=0)
    average_position_distance = np.linalg.norm(average_position)
    return {"average_distance": average_position_distance}


def evaluate_average_distance_to_prey(data) -> dict:
    """
    Evaluate the average position of all robots.

    :param data: dictionary containing information about entities, including their final positions.
    :return: A dictionary containing:
        - average_position (tuple): The average position of all robots.
    """
    robot_positions_final = []
    prey_final_position = []
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions_final.append(info["trajectory"][-1])
        elif info["type"] == "Prey":
            prey_final_position = info["trajectory"][-1]

    robot_positions_final = np.array(robot_positions_final)
    num_robots = len(robot_positions_final)

    if num_robots == 0:
        raise ValueError("No robots found in the data.")
    average_position = np.mean(robot_positions_final, axis=0)
    average_position_distance = np.linalg.norm(average_position - prey_final_position)
    return {"average_distance": average_position_distance}


def evaluate_shape_similarity(data, target_shape: list) -> dict:
    """
    Evaluate the similarity between the robots' final positions and a specified shape using Procrustes analysis.

    :param data: dictionary containing information about entities, including their trajectories.
    :param target_shape: a list of tuples representing the vertices of the target shape in order.
    :return: A dictionary containing:
        - procrustes_distance (float): A measure of dissimilarity between the robot positions and the target shape using Procrustes analysis.
    """
    robot_positions_final = []

    # Collect the final positions of the robots
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions_final.append(info["trajectory"][-1])

    robot_positions_final = np.array(robot_positions_final)

    if len(robot_positions_final) == 0:
        raise ValueError("No robots found in the data.")

    # Step 1: Convert the target shape to a numpy array
    target_shape = np.array(target_shape)

    # Step 2: Sort robot positions based on the closest points in the target shape
    if len(robot_positions_final) != len(target_shape):
        raise ValueError(
            "The number of robot positions and the target shape points must be equal for Procrustes analysis."
        )

    # Calculate the distance matrix between robot final positions and target shape points
    distances = distance_matrix(robot_positions_final, target_shape)

    # Step 3: Match each robot's final position to the closest target shape point
    sorted_robot_positions = []
    sorted_target_shape = []
    used_robot_indices = set()

    for i in range(len(target_shape)):
        # Find the closest robot to the current target point
        closest_robot_index = np.argmin(distances[:, i])

        # Ensure that no robot is used more than once
        while closest_robot_index in used_robot_indices:
            distances[closest_robot_index, i] = np.inf
            closest_robot_index = np.argmin(distances[:, i])

        used_robot_indices.add(closest_robot_index)
        sorted_robot_positions.append(robot_positions_final[closest_robot_index])
        sorted_target_shape.append(target_shape[i])

    # Convert sorted lists to numpy arrays for Procrustes analysis
    sorted_robot_positions = np.array(sorted_robot_positions)
    sorted_target_shape = np.array(sorted_target_shape)

    # Step 4: Perform Procrustes analysis
    _, _, disparity = procrustes(sorted_target_shape, sorted_robot_positions)

    return {"procrustes_distance": disparity}


def evaluate_encircling_end(
        data, target_radius: float = 1, tolerance: float = 0.1
) -> dict:
    """
    Evaluate the task completion by calculating the mean and variance of the robots' final distances to the prey,
    and how well they match the target radius at the final step.

    :param data: dictionary containing information about entities, including their trajectories.
    :param target_radius: the target distance from the prey's position (default is 0.5).
    :param tolerance: the distance tolerance within which the target is considered achieved.
    :return: A dictionary containing:
        - mean_distance_error (float): The mean absolute error of the robots' final distance to the target radius.
        - variance_distance_error (float): The variance of the distance errors.
        - initial_distance_error (float): The initial mean distance to the target radius.
    """
    robot_positions_initial = []
    robot_positions_final = []
    initial_distances = []
    final_distances = []

    prey_trajectory = None

    # First pass: find the prey trajectory
    for entity_id, info in data.items():
        if info["type"] == "Prey":
            prey_trajectory = np.array(info["trajectory"])
            break

    if prey_trajectory is None:
        raise ValueError("Prey trajectory not found in data.")

    # Get the final position of the prey
    prey_final_position = prey_trajectory[-1]

    # Process robot data
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            trajectory = np.array(info["trajectory"])
            initial_distances.append(
                np.linalg.norm(trajectory[0] - prey_trajectory[0])
            )  # Distance at the start
            robot_positions_initial.append(trajectory[0])
            robot_positions_final.append(trajectory[-1])  # Distance at the final step

    robot_positions_final = np.array(robot_positions_final)

    # Calculate final distances between robots and prey
    final_distances = np.linalg.norm(
        robot_positions_final - prey_final_position, axis=1
    )

    # Mean distance error at the final step
    mean_distance_error = np.mean(np.abs(final_distances - target_radius))

    # Variance of the distance errors at the final step
    variance_distance_error = np.var(np.abs(final_distances - target_radius))

    # Initial mean distance
    initial_mean_distance = np.mean(initial_distances)

    return {
        "mean_distance_error": mean_distance_error,
        "variance_distance_error": variance_distance_error,
        "initial_distance_error": abs(initial_mean_distance - target_radius),
    }


def evaluate_robot_quadrant_positions(
        data, target_regions: dict, tolerance: float = 0.1
) -> dict:
    """
    根据机器人最开始所在的象限进行分类，并检查每一类机器人是否最终位于指定的区域内。

    :param data: 包含实体信息的字典，包括其类型和轨迹。
    :param target_regions: 一个字典，键为象限编号，值为每个象限对应的最终目标区域，格式为(x_min, x_max, y_min, y_max)。
    :param tolerance: 机器人最终位置与目标区域的容差，默认值为0.1。
    :return: 一个字典，包含：
        - achieved_robots_by_quadrant (dict): 每个象限中符合要求的机器人数量。
        - total_achieved (int): 符合要求的机器人总数。
        - achievement_ratio (float): 符合要求的机器人占总数的比例。
    """
    quadrant_robots = {1: [], 2: [], 3: [], 4: []}  # 第一象限  # 第二象限  # 第三象限  # 第四象限

    # 根据初始位置对机器人进行象限分类
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            initial_position = np.array(info["trajectory"][0])
            if initial_position[0] >= 0 and initial_position[1] >= 0:
                quadrant_robots[1].append((entity_id, info))
            elif initial_position[0] < 0 and initial_position[1] >= 0:
                quadrant_robots[2].append((entity_id, info))
            elif initial_position[0] < 0 and initial_position[1] < 0:
                quadrant_robots[3].append((entity_id, info))
            elif initial_position[0] >= 0 and initial_position[1] < 0:
                quadrant_robots[4].append((entity_id, info))

    # 计算符合要求的机器人数量
    achieved_robots_by_quadrant = {1: 0, 2: 0, 3: 0, 4: 0}
    total_robots = sum(len(robots) for robots in quadrant_robots.values())

    for quadrant, robots in quadrant_robots.items():
        target_region = target_regions.get(quadrant)
        if target_region is None:
            continue

        x_min, x_max, y_min, y_max = target_region

        for entity_id, info in robots:
            final_position = np.array(info["trajectory"][-1])
            if (
                    x_min - tolerance <= final_position[0] <= x_max + tolerance
                    and y_min - tolerance <= final_position[1] <= y_max + tolerance
            ):
                achieved_robots_by_quadrant[quadrant] += 1

    total_achieved = sum(achieved_robots_by_quadrant.values())
    achievement_ratio = total_achieved / total_robots if total_robots > 0 else 0

    return {
        "achieved_robots_by_quadrant": achieved_robots_by_quadrant,
        "total_achieved": total_achieved,
        "achievement_ratio": achievement_ratio,
    }


def evaluate_robot_prey_distance(
        data, distance_threshold: float = 1.0, proportion_threshold: float = 0.8
) -> dict:
    """
    计算每个时刻机器人与猎物之间的距离，并统计至少80%机器人在猎物1m范围内的step数占总步数的比例。

    :param data: 包含实体信息的字典，包括其轨迹。
    :param distance_threshold: 机器人与猎物的最大允许距离（默认是1m）。
    :param proportion_threshold: 满足条件的机器人比例阈值（默认是80%）。
    :return: 一个字典，包含：
        - within_distance_steps_ratio (float): 满足条件的步数占总步数的比例。
        - total_steps (int): 总的时间步数。
        - within_distance_steps (int): 满足条件的时间步数。
    """
    robot_positions = []
    prey_trajectory = None

    # 获取猎物的轨迹
    for entity_id, info in data.items():
        if info["type"] == "Prey":
            prey_trajectory = np.array(info["trajectory"])
            break

    if prey_trajectory is None:
        raise ValueError("Prey trajectory not found in data.")

    # 获取所有机器人的轨迹
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            robot_positions.append(np.array(info["trajectory"]))

    if len(robot_positions) == 0:
        raise ValueError("No robots found in data.")

    robot_positions = np.array(robot_positions)
    num_robots = robot_positions.shape[0]
    total_steps = len(prey_trajectory)
    within_distance_steps = 0

    # 遍历每个时间步
    for t in range(total_steps):
        distances = np.linalg.norm(
            robot_positions[:, t, :] - prey_trajectory[t], axis=1
        )
        # 统计距离猎物在指定范围内的机器人数量
        num_within_distance = np.sum(distances <= distance_threshold)

        # 如果至少80%的机器人在指定范围内
        if num_within_distance / num_robots >= proportion_threshold:
            within_distance_steps += 1

    within_distance_steps_ratio = (
        within_distance_steps / total_steps if total_steps > 0 else 0
    )

    return {
        "within_distance_steps_ratio": within_distance_steps_ratio,
        "total_steps": total_steps,
        "within_distance_steps": within_distance_steps,
    }


def check_robots_no_movement_in_first_third(data) -> dict:
    """
    检查所有机器人在前1/3时间内是否有移动，并返回未移动的机器人ID。

    :param data: 包含实体信息的字典，包括其类型和轨迹。
    :return: 未在前1/3时间内移动的机器人ID列表。
    """
    no_movement_robots = []
    if not data:
        return []
    for entity_id, info in data.items():
        if info["type"] == "Robot":
            trajectory = np.array(info["trajectory"])
            num_timesteps = len(trajectory)
            one_third_timestep = num_timesteps // 3

            # 判断机器人是否在前1/3时间内移动
            initial_position = trajectory[10]
            has_moved = False

            for t in range(1, one_third_timestep):
                if not np.allclose(initial_position, trajectory[t]):
                    has_moved = True
                    break

            if not has_moved:
                no_movement_robots.append(entity_id)

    return {"no_move_num": len(no_movement_robots)}

# if __name__ == '__main__':
#     _, _, disparity = procrustes(target_shape, robot_positions_final)
