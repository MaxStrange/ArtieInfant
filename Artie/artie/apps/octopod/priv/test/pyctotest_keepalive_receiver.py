import pyctopod

def handle_message(from_id, topic, msg):
    """
    Publishes "blah" to :test everytime a message is received.
    """
    msg = "blah"
    pyctopod.publish("test", msg.encode('utf8'), from_id="pyctotest_keepalive_receiver")

def main():
    pyctopod.subscribe(topics=["test_topic"], handlers=[handle_message])

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)

