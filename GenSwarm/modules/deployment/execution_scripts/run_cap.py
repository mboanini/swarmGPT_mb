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

import os
import pickle
import sys
import threading
from xmlrpc.client import escape

import rospy
from code_llm.srv import GetTargetPositions, GetCharPoints, GetCharPointsRequest


def get_target_positions():
    rospy.wait_for_service("/get_target_positions")
    try:
        get_target_positions = rospy.ServiceProxy(
            "/get_target_positions", GetTargetPositions
        )
        response = get_target_positions()
        return {
            info.id: (info.position.x, info.position.y)
            for info in response.target_positions
        }
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return {}


def get_contour_points(character):
    rospy.wait_for_service("/get_char_points")
    try:
        get_char_points = rospy.ServiceProxy("/get_char_points", GetCharPoints)
        request = GetCharPointsRequest(character=character)
        response = get_char_points(request)
        points = [(point.x, point.y) for point in response.points]
        return points
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return []


def run_robot(robot_id, target_position, formation_points, task=None):
    from api import initialize_ros_node, init_node
    init_node()
    initialize_ros_node(
        robot_id=robot_id,
        target_position=target_position,
        formation_points=formation_points,
        assigned_task=task,
    )
    from main import main
    main()


def run_multiple_robot(start_idx, end_idx):
    rospy.init_node(f"multi_robot_publisher_node{start_idx}_{end_idx}", anonymous=True)
    target_positions = get_target_positions()
    # char_points = get_contour_points('R')
    char_points = [(1, -1), (1, 1), (0, 0), (1, 0), (2, 0)]
    threads = []

    for i in range(start_idx, end_idx + 1):
        thread = threading.Thread(
            target=run_robot, args=(i, target_positions[i], char_points, None)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    a = sys.argv
    # start_id = 1
    # end_id = 9
    start_id = int(sys.argv[1])
    end_id = int(sys.argv[2])

    run_multiple_robot(start_id, end_id)
