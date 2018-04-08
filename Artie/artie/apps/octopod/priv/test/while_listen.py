"""
This test registers with erlport so that it saves received messages to disk
as 'savefiletest<X>.wav' where <X> starts at 0 and increments once each time
a new message is received.
"""
from erlport.erlterms import Atom
import erlport.erlang as erl

nextnum = 0

def register_handler():
    def handler(msg):
        nextpath = "savefiletest" + str(nextnum) + ".wav"
        with open(nextpath, 'wb') as f:
            f.write(msg)
        global nextnum
        nextnum += 1

    erl.set_message_handler(handler)
    return Atom("ok")

def main():
    register_handler()

    s = 0
    while True:
        time.sleep(100)
        s += 1
    return s
