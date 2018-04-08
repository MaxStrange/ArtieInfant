"""
This test script does nothing in a while loop, but registers a message handler with
erlport so that when it receives a message, it appends " FROM PYTHON!" to it and
sends it back.
"""
from erlport.erlterms import Atom
import erlport.erlang as erl
import time

def register_handler(pid):
    def handler(msg):
        erl.cast(pid, msg)
    erl.set_message_handler(handler)
    return Atom("ok")

def main(pid):
    s = 0
    while True:
        time.sleep(100)
        s += 1
    return s
