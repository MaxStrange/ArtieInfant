import pyctopod

def main():
    pyctopod.publish("test_topic", "This is a Test".encode('utf8'), from_id="pyctotest_pub_one_msg")

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)

