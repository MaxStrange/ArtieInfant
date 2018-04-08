import threading
import time
import sys

from erlport.erlang import set_message_handler, cast
from erlport.erlterms import Atom

message_handler = None
filenum = 0

def register_handler(pid):
    global message_handler
    message_handler = pid

def save_file(fcontents, fname):
  """
  Saves the given fcontents to file named fname.
  """
  with open(fname, 'wb') as f:
      f.write(fcontents)

def handle_message(msg):
    """
    msg is the contents of a file. We save the file as 'saved_file<X>.wav'
    where <X> is 0 to start with, and increments one each time the
    handler receives a message.
    """
    global filenum
    newname = "saved_file" + str(filenum) + ".wav"
    threading.Thread(target=save_file, args=(msg, newname)).start()
    #with open(newname, 'wb') as f:
    #    f.write(msg)
    filenum += 1

    from_atom = Atom("pyprocess".encode('utf8'))
    ok_atom = Atom("ok".encode('utf8'))
    cast(message_handler, (from_atom, ok_atom))

set_message_handler(handle_message)

