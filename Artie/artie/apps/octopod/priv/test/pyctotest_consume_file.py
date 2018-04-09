import pyctopod
import random
import string

def handle_message(from_id, topic, msg):
    """
    Writes `msg` to file and publishes the filepath to :test.
    """
    randstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    fpath = "furelise" + str(randstring) + ".wav"
    with open(fpath, 'wb') as f:
        f.write(msg)
    pyctopod.publish("test", fpath.encode('utf8'), from_id="pyctotest_consume_file")

def main():
    pyctopod.subscribe(topics=["test_topic"], handlers=[handle_message])

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)

