import time
import sys

from erlport.erlang import set_message_handler, cast
from erlport.erlterms import Atom

message_handler = None

def register_handler(pid):
    global message_handler
    message_handler = pid

def handle_message(msg):
    result = msg.decode('utf8') + " FROM PYTHON!"
    try:
        atom = Atom("ok".encode('utf8'))
    except Exception as e:
        print("Could not convert 'python' to atom.")
    try:
        cast(message_handler, (atom, result.encode('utf8')))
    except Exception as e:
        print("Could not cast.")

set_message_handler(handle_message)

