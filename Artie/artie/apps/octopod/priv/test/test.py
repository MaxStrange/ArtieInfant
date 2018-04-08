import time
import sys

from erlport.erlang import set_message_handler, cast
from erlport.erlterms import Atom

message_handler = None

def cast_message(pid, message):
    cast(pid, message)

def register_handler(pid):
    global message_handler
    message_handler = pid

def handle_message(count):
    try:
        print("Received message from Elixir")
        print(count)
        result = long_counter(count)
        if message_handler:
            cast_message(message_handler, (Atom('python'), result))
    except Exception as e:
        pass

def long_counter(count=100):
    i = 0
    data = []
    while i < count:
        time.sleep(1)
        data.append(i+1)
        i = i + 1
    return data

set_message_handler(handle_message)
