import time
import sys

from erlport.erlang import set_message_handler, cast
from erlport.erlterms import Atom

message_handler = None
filenum = 0

def register_handler(pid):
    global message_handler
    message_handler = pid

def handle_message(msg):
    """
    msg is the contents of a file. We save the file as 'saved_file<X>.wav'
    where <X> is 0 to start with, and increments one each time the
    handler receives a message.
    """
    print("Got message from elixir")
    global filenum
    newname = "saved_file" + str(filenum) + ".wav"
    print("saving to", newname)
    with open(newname, 'wb') as f:
        f.write(msg)
    filenum += 1

    from_atom = Atom("pyprocess")
    ok_atom = Atom("ok".encode('utf8'))
    cast(message_handler, (from_atom, ok_atom))

set_message_handler(handle_message)

