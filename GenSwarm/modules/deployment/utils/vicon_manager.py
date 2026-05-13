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

#!/usr/bin/env python

import rospy
from subprocess import check_output
from geometry_msgs.msg import PoseStamped, TwistStamped
import json
import re
import os

data = {
    "entities": {
        "robot": {"specified": []},
        "leader": {"specified": []},
        "obstacle": {"specified": []},
        "landmark": {"specified": []},
        "pushable_object": {"specified": []},
    }
}
data_template = {"id": 0, "position": [0, 0], "size": 0.15, "velocity": [0, 0]}

entity_type_list = ["VSWARM", "OBSTACLE"]
entity_type_mapping = {"VSWARM": "robot", "OBSTACLE": "obstacle"}

subscribers = []


def update_entity_data(entity_id, entity_type, key, value):
    mapped_type = entity_type_mapping.get(entity_type, entity_type)

    if mapped_type not in data["entities"]:
        data["entities"][mapped_type] = {"specified": []}

    entity_list = data["entities"][mapped_type]["specified"]
    entity = next((item for item in entity_list if item["id"] == entity_id), None)
    if not entity:
        entity = data_template.copy()
        entity["id"] = entity_id
        entity_list.append(entity)

    entity[key] = value


def pose_callback(msg, args):
    entity_id, entity_type = args
    position = [msg.pose.position.x, msg.pose.position.y]
    entity_id = int(entity_id)
    update_entity_data(entity_id, entity_type, "position", position)


def twist_callback(msg, args):
    entity_id, entity_type = args
    velocity = [msg.twist.linear.x, msg.twist.linear.y]
    entity_id = int(entity_id)
    update_entity_data(entity_id, entity_type, "velocity", velocity)


def save_data_to_json():
    file_path = os.path.join(
        os.path.dirname(__file__), "../../../config/vicon_data.json"
    )
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print("Data saved to JSON.")


def get_entity_ids(entity_prefix):
    topic_list = (
        check_output(["/opt/ros/noetic/bin/rostopic", "list"])
        .decode("utf-8")
        .split("\n")
    )
    print(f"Available topics: {topic_list}")
    entity_ids = set()
    for topic in topic_list:
        match = re.search(rf"/vrpn_client_node/{entity_prefix}(\d+)/", topic)
        if match:
            entity_ids.add(match.group(1))
    return list(entity_ids)


def generate_subscribers(entity_id, entity_type):
    pose_topic = f"/vrpn_client_node/{entity_type.upper()}{entity_id}/pose"
    twist_topic = f"/vrpn_client_node/{entity_type.upper()}{entity_id}/twist"

    print(f"Subscribing to {pose_topic} and {twist_topic}")

    subscribers.append(
        rospy.Subscriber(
            pose_topic,
            PoseStamped,
            pose_callback,
            callback_args=(entity_id, entity_type),
        )
    )
    subscribers.append(
        rospy.Subscriber(
            twist_topic,
            TwistStamped,
            twist_callback,
            callback_args=(entity_id, entity_type),
        )
    )
    pose_msg = rospy.wait_for_message(pose_topic, PoseStamped)
    twist_msg = rospy.wait_for_message(twist_topic, TwistStamped)
    print(f"Received first message for {entity_id} of type {entity_type}")
    pose_callback(pose_msg, (entity_id, entity_type))
    twist_callback(twist_msg, (entity_id, entity_type))
    print(f"Received first message for {entity_id} of type {entity_type}")


def listener():
    rospy.init_node("dynamic_topic_subscriber", anonymous=True)
    global entity_type_list
    for entity_type in entity_type_list:
        ids = get_entity_ids(entity_type)
        print(f"Found IDs for {entity_type}: {ids}")
        for entity_id in ids:
            generate_subscribers(entity_id, entity_type)
    print("All subscribers initialized and received first message.")
    save_data_to_json()
    rospy.signal_shutdown("Data saved to JSON.")
    rospy.spin()


if __name__ == "__main__":
    listener()
