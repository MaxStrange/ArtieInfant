import pyctopod

def handle_message(from_id, topic, msg):
    """
    Publishes `msg + " FROM PYTHON!"` to :test.
    """
    msg = msg.decode('utf8') + " FROM PYTHON!"
    pyctopod.publish("test", msg, from_id="pyctotest_consume")

def main():
    pyctopod.subscribe(topics=["test_topic"], handlers=[handle_message])

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)

