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

import rospy
from geometry_msgs.msg import Twist
import time


class VelocityListener:
    def __init__(self):
        self.subscribers = []
        self.start_time = None
        self.total_time = 0.0
        self.message_count = 0
        self.topic_update_count = {}
        self.all_topics_updated = False

        self.test_results = []
        self.test_count = 10

        robot_start_index = rospy.get_param("robot_start_index")
        robot_end_index = rospy.get_param("robot_end_index")
        for i in range(robot_start_index, robot_end_index + 1):
            topic_name = f"/robot_{i}/velocity"
            self.topic_update_count[topic_name] = 0
            rospy.Subscriber(
                topic_name, Twist, self.velocity_callback, callback_args=topic_name
            )

    def velocity_callback(self, data: Twist, topic_name):
        if self.start_time is None:
            self.start_time = time.time()

        self.message_count += 1
        self.topic_update_count[topic_name] += 1

        elapsed_time = time.time() - self.start_time
        self.total_time += elapsed_time
        self.start_time = time.time()

        if (
            all(count > 0 for count in self.topic_update_count.values())
            and not self.all_topics_updated
        ):
            self.all_topics_updated = True
            self.print_statistics_and_prepare_for_next_test()

    def print_statistics_and_prepare_for_next_test(self):
        sorted_topics = sorted(
            self.topic_update_count.items(), key=lambda item: item[1], reverse=True
        )
        print(
            f"Message count: {self.message_count}, Total time: {self.total_time:.4f} seconds"
        )
        for topic, count in sorted_topics:
            print(f"{topic}: {count} messages")

        self.test_results.append(self.total_time)
        self.prepare_for_next_test()

    def prepare_for_next_test(self):
        # Reset the state for the next test
        self.start_time = None
        self.total_time = 0.0
        self.message_count = 0
        self.topic_update_count = {topic: 0 for topic in self.topic_update_count}
        self.all_topics_updated = False

        if len(self.test_results) < self.test_count:
            print(
                f"Preparing for next test..., {len(self.test_results) + 1}/{self.test_count}"
            )
        else:
            self.print_final_results()
            rospy.signal_shutdown("All tests completed.")

    def start_tests(self):
        while not rospy.is_shutdown() and len(self.test_results) < self.test_count:
            self.start_time = None
            self.total_time = 0.0
            self.message_count = 0
            self.all_topics_updated = False
            rospy.sleep(1)  # Add a small delay between tests to avoid conflicts

            rate = rospy.Rate(100)  # 10 Hz
            while not rospy.is_shutdown() and not self.all_topics_updated:
                rate.sleep()

    def print_final_results(self):
        print("\nTest results:")
        for i, result in enumerate(self.test_results):
            print(f"Test {i + 1}: {result:.4f} seconds")

        average_time = sum(self.test_results) / len(self.test_results)
        print(f"\nAverage time: {average_time:.4f} seconds")


if __name__ == "__main__":
    rospy.init_node("velocity_listener", anonymous=True)
    listener = VelocityListener()
    listener.start_tests()
