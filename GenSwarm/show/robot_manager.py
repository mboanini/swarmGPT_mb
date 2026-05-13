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

import json

from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
import numpy as np
import socket
import threading
import time
import rospy
import os
from paho.mqtt import client as mqtt_client
from rospy_message_converter import json_message_converter
from sympy.physics.units.definitions.dimension_definitions import angle
from tensorflow.python.distribute.multi_process_runner import manager
from scipy.spatial.transform import Rotation as R


class OmniEngine:
    def __init__(self):
        super().__init__()
        self.subscribers = []
        self.agent_list = [0, 1, 2, 3, 4]
        # self.agent_list = [ 4]
        self.agents_num = len(self.agent_list)
        self.positions = {}
        self.angles = {}
        self.generate_all_subscribers()

    def pose_callback(self, msg, args):
        print("pose call back")
        try:
            entity_id = args
            position = np.array([msg.pose.position.x, msg.pose.position.y])
            quaternion = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            )
            rot_mat = R.from_quat(quaternion).as_matrix()
            # 解出欧拉角（RPY顺序）
            self.euler = np.array(R.from_matrix(rot_mat).as_euler("xyz", degrees=False))
            self.angles[entity_id] = self.euler[2]
            self.positions[entity_id] = position
            print(f"update position of omni bot {entity_id} to {position}")

        except Exception as e:
            print(f"Error in pose callback: {e}")
        # print(f"euler of entity {entity_id} is {self.euler}")

        # print(f"update position of omni bot {entity_id} to {position}")

    def generate_subscribers(self, entity_id):
        pose_topic = f"/vrpn_client_node/VSWARM{entity_id}/pose"

        self.subscribers.append(
            rospy.Subscriber(
                pose_topic, PoseStamped, self.pose_callback, callback_args=(entity_id)
            )
        )
        print(f"waiting for message{pose_topic}")
        msg = rospy.wait_for_message(pose_topic, PoseStamped)
        position = np.array([msg.pose.position.x, msg.pose.position.y])
        self.positions[entity_id] = position

    def generate_all_subscribers(self):
        for i in self.agent_list:
            self.generate_subscribers(entity_id=i)

    def get_position(self, entity_id):
        return self.positions[entity_id]

    def get_angle(self, entity_id):
        return self.angles[entity_id]


class MqttClientThread:
    def __init__(self, broker, port, keepalive, client_id):
        self.broker = broker  # MQTT代理服务器地址
        self.port = port
        self.keepalive = keepalive
        self.client_id = client_id
        self.client = self.connect_mqtt()
        self.client.loop_start()

    def connect_mqtt(self):
        """连接MQTT代理服务器"""

        def on_connect(client, userdata, flags, rc):
            """连接回调函数"""
            if rc == 0:
                print("Connected to MQTT OK!")
            else:
                print(f"Failed to connect, return code {rc}")

        client = mqtt_client.Client(self.client_id)
        client.on_connect = on_connect
        client.connect(self.broker, self.port, self.keepalive)
        return client

    def publish(self, topic, msg):
        """发布消息到指定主题"""
        result = self.client.publish(topic, msg)
        status = result[0]
        if status == 0:
            print(f"Sent `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")

    def run(self):
        """启动MQTT客户端"""
        try:
            self.client.loop_forever()
        except Exception as e:
            print(f"Error in MQTT loop: {e}")


class Manager:
    def __init__(self):
        self.omni_engine = OmniEngine()
        self.mqtt_client = self.start_up_mqtt_thread()

    def start_up_mqtt_thread(self):
        broker_ip = "10.0.2.66"
        port = 1883
        keepalive = 60  # 与代理通信之间允许的最长时间段（以秒为单位）
        client_id = f"{socket.gethostname()}_robot_pub"  # 客户端id不能重复

        try:
            broker = os.environ["REMOTE_SERVER"]
        except KeyError:
            broker = broker_ip

        net_status = -1
        while net_status != 0:
            net_status = os.system(f"ping -c 4 {broker}")
            time.sleep(2)

        # 启动MQTT客户端线程
        mqtt_client_instance = MqttClientThread(
            broker=broker, port=port, keepalive=keepalive, client_id=client_id
        )
        mqtt_thread = threading.Thread(target=mqtt_client_instance.run)
        mqtt_thread.start()
        return mqtt_client_instance

    def get_position(self, entity_id):
        position = self.omni_engine.get_position(entity_id)
        return position

    def get_euler(self, entity_id):
        euler = self.omni_engine.get_angle(entity_id)
        print(f"get euler of entity {entity_id} to {euler}")
        return euler

    def set_position(self, entity_id, position):
        json_msg = {
            "cmd_type": "pos_ctrl",
            "args": {"0": position[0], "1": position[1]},
        }
        print(json_msg)
        json_str = json.dumps(json_msg)
        self.mqtt_client.publish(
            f"/VSWARM{entity_id}_robot/cmd", json_str.encode("utf-8")
        )

    def set_velocity(self, entity_id, velocity):
        # msg = Twist()
        # msg.linear.x = velocity[0]
        # msg.linear.y = velocity[1]
        # json_str = json_message_converter.convert_ros_message_to_json(msg)
        json_msg = {"x": velocity[0], "y": velocity[1], "theta": velocity[2]}
        if entity_id == 0:
            print(f"set velocity of entity {entity_id} to {velocity}")
        json_str = json.dumps(json_msg)
        self.mqtt_client.publish(
            f"/VSWARM{entity_id}_robot/motion", json_str.encode("utf-8")
        )


rospy.init_node("multi_robot_controller")

manager_instance = Manager()
