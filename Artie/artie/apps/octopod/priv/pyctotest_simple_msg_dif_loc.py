import pyctopod
import time

def publish_simple_message(topic, msg):
    """
    Publishes `msg` to `topic`.
    """
    pyctopod.publish(topic, msg, from_id="tester")

def node_main():
    """
    Calls this function when Elixir is ready.
    """
    publish_simple_message("test_topic", "This is a Test".encode('utf8'))

def register_handler(pid):
    pyctopod.register_main(node_main)
    pyctopod.register_handler(pid)

