"""
This is the python module that handles all the Elixir-Python
interfacing. Client python modules should never need
to reference erlport at all and should instead, handle all the
publisher/subscriber interactions through this module.
"""
import queue
import threading
from erlport.erlang import set_message_handler, cast
from erlport.erlterms import Atom

_msg_handling_pid = None
_msgq = queue.Queue()

## Client-Facing API

def init():
    pass

def subscribe(topics, handlers):
    pass

def publish(topics, msg):
    pass

## Erlport API: Don't use this in client modules

def register_handler(pid):
    """
    Registers the given Elixir process as the handler
    for this library.
    """
    global _msg_handling_pid
    _msg_handling_pid = pid

def _handle_message(msg):
    """
    `msg` should be a tuple of the form: (topic, payload).

    Calls the correct handler for the topic.
    """
    print(msg.decode('utf8'))

# Register the handler function with Elixir
set_message_handler(_handle_message)

