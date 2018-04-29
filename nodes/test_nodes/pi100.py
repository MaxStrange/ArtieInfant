import pyctopod

def main():
    print("HEY GIRL")
    for i in range(1000):
        pyctopod.publish("test_topic", "pi100".encode('utf8'), from_id="pi100")

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)
