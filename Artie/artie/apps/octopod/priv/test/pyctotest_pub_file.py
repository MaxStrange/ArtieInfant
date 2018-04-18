import pyctopod

def main():
    with open("furelise.wav", 'rb') as f:
        contents = f.read()
    pyctopod.publish("test_topic", contents, from_id="pyctotest_pub_file")

def register_handler(pid):
    pyctopod.register_main(main)
    pyctopod.register_handler(pid)

