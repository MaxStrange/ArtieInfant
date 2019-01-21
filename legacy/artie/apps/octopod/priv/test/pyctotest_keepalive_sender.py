import pyctopod
import time

def main():
    time.sleep(5)
    pyctopod.publish("test_topic", "stay_alive".encode('utf8'), from_id="pyctotest_keepalive_sender")
    time.sleep(35)
    pyctopod.publish("test_topic", "stay_alive".encode('utf8'), from_id="pyctotest_keepalive_sender")
    time.sleep(45)
    pyctopod.publish("test_topic", "stay_alive".encode('utf8'), from_id="pyctotest_keepalive_sender")

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)

