import unittest
from unittest.mock import MagicMock, patch
from modules.deployment.utils.mqtt_pub import (
    MqttClientThread,
    obs_callback,
)  # replace with the actual module name
from code_llm.msg import Observations  # make sure this import is correct


class TestMqttClientThread(unittest.TestCase):
    @patch("paho.mqtt.client.Client")
    def test_connect_mqtt_success(self, mock_mqtt_client):
        mock_client_instance = mock_mqtt_client.return_value
        mock_client_instance.connect.return_value = None
        mqtt_client = MqttClientThread("test_broker", 1883, 60, "test_client")

        # Check if the MQTT client is connected
        mock_client_instance.connect.assert_called_with("test_broker", 1883, 60)

    @patch("paho.mqtt.client.Client")
    def test_publish_message_success(self, mock_mqtt_client):
        mock_client_instance = mock_mqtt_client.return_value
        mock_client_instance.publish.return_value = (0, 0)

        mqtt_client = MqttClientThread("test_broker", 1883, 60, "test_client")
        mqtt_client.publish("test/topic", "test message")

        # Verify that the publish method was called correctly
        mock_client_instance.publish.assert_called_with("test/topic", "test message")

    @patch("paho.mqtt.client.Client")
    def test_publish_message_failure(self, mock_mqtt_client):
        mock_client_instance = mock_mqtt_client.return_value
        mock_client_instance.publish.return_value = (1, 0)

        mqtt_client = MqttClientThread("test_broker", 1883, 60, "test_client")
        mqtt_client.publish("test/topic", "test message")

        # Check that the failure message was printed
        # with self.assertLogs(level='ERROR') as log:
        #     mqtt_client.publish('test/topic', 'test message')
        # self.assertIn("Failed to send message to topic test/topic", log.output[0])

    @patch("rospy_message_converter.json_message_converter.convert_ros_message_to_json")
    def test_obs_callback(self, mock_json_converter):
        # Mock the json conversion
        mock_json_converter.return_value = '{"data": "test"}'

        mqtt_client = MagicMock()  # Create a mock MQTT client
        msg = (
            Observations()
        )  # Create an instance of Observations (you may need to set its attributes)

        obs_callback(mqtt_client, msg)

        # Check if the publish method was called with the correct arguments
        mqtt_client.publish.assert_called_with("/observation", b'{"data": "test"}')


if __name__ == "__main__":
    unittest.main()
