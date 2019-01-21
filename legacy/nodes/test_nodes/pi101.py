import pyctopod

def handle_message(from_id, topic, msg):
    print("GOT A MESSAGE")
    msg = msg.decode('utf8') + " pi101"
    pyctopod.publish("eavesdrop", msg.encode('utf8'), from_id="pi101")

def main():
    pyctopod.subscribe(topics=["test_topic"], handlers=[handle_message])

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)
