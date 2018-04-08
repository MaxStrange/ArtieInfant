import queue
import threading
import time
import sys

from erlport.erlang import set_message_handler, cast
from erlport.erlterms import Atom

message_handler = None
filenum = 0
msgq = queue.Queue()

def register_handler(pid):
    global message_handler
    message_handler = pid

def save_file(fcontents, fnum, msgq):
    """
    Saves the given fcontents to file named fname.
    """
    fname = "saved_file" + str(fnum) + ".wav"
    with open(fname, 'wb') as f:
        f.write(fcontents)

    # Signal Elixir that we are done
    msgq.put(fnum)

def alert_elixir(msgq):
    """
    Waits around on the message queue and sends
    {:pyprocess, :msgq.get()} to the erlport.
    """
    from_atom = Atom("pyprocess".encode('utf8'))
    while True:
        try:
            msg = msgq.get(timeout=10)
            cast(message_handler, (from_atom, msg))
        except queue.Empty:
            return  # Probably the elixir process is dead, so we should end


def handle_message(msg):
    """
    msg is the contents of a file. We save the file as 'saved_file<X>.wav'
    where <X> is 0 to start with, and increments one each time the
    handler receives a message.
    """
    global filenum
    threading.Thread(target=save_file, args=(msg, filenum, msgq)).start()
    filenum += 1

threading.Thread(target=alert_elixir, args=(msgq,)).start()
set_message_handler(handle_message)

