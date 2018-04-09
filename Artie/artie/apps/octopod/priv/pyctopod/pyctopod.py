"""
This is the python module that handles all the Elixir-Python
interfacing. Client python modules should never need
to reference erlport at all and should instead, handle all the
publisher/subscriber interactions through this module.
"""
import queue
import threading
import time
from erlport.erlang import set_message_handler, cast
from erlport.erlterms import Atom

## A horrible number of globals... Ugh.
_msg_handling_pid = None
_msgq = queue.Queue()
_main_func = None
_topic_handlers = {}
_consumption_thread = None


## Client-Facing API

def register_main(func):
    """
    Registers the main function to execute - will execute this
    upon receipt of a signal from Elixir that everything is set
    up and ready to go. Will run it in a separate thread.
    """
    global _main_func
    _main_func = func

def register_handler(pid):
    """
    Registers the given Elixir process as the handler
    for this library.

    This function must be called first - and the client
    module must define a function with the same prototype which
    simply calls this function.
    """
    global _msg_handling_pid
    _msg_handling_pid = pid

def subscribe(topics, handlers):
    """
    Subscribes to each topic in `topics` (which may be a single
    str). Whenever a message is received from a topic, the
    associated handler is called - the handlers are associated
    with the topics based purely on order: topic0 -> handler0.

    The handlers are functions that take 'from_id', 'topic', msg.
    The handlers are called asynchronously, two messages received
    on the same topic will both fire without having to finish one.
    """
    if type(topics) == str:
        topics = [topics]

    global _topic_handlers
    for topic, handler in zip(topics, handlers):
        _topic_handlers[topic] = handler

    global _consumption_thread
    if _consumption_thread is None:
        _consumption_thread = threading.Thread(target=_consume)
        _consumption_thread.start()

def publish(topics, msg, from_id='default'):
    """
    Publishes `msg` to all topics in `topics`, which may be a
    single topic.

    Message must be bytes. Topics must be a string. From_id must
    also be a string.
    """
    if type(msg) != bytes:
        raise TypeError("msg must be of type 'bytes' but is of type " + str(type(msg)))

    if type(topics) == str:
        topics = [topics]
    try:
        topics = [Atom(t.encode('utf8')) for t in topics]
    except TypeError:
        topics = [Atom(topics.encode('utf8'))]

    id_as_atom = Atom(from_id.encode('utf8'))
    for topic in topics:
        cast(_msg_handling_pid, (id_as_atom, topic, msg))


def _consume():
    """
    Sits around waiting for messages from Elixir. Expects a
    keepalive message to the :priv_keepalive topic at least
    once every a minute, otherwise exits.

    Spawns a new thread every time a new message is received
    on a topic that has a handler associated with it.
    """
    keepalive_interval = 30
    start = time.time()
    while True:
        if time.time() - start > keepalive_interval:
            return
        try:
            from_id, topic, msg = _msgq.get(timeout=keepalive_interval)
            if topic == "priv_keepalive":
                start = time.time()
            else:
                handler = _topic_handlers[topic]
                threading.Thread(target=handler, args=(from_id, topic, msg)).start()
        except queue.Empty:
            return
        except KeyError as e:
            print(e)
            print("Could not find key {} in {}".format(topic, _topic_handlers))
            raise


## Erlport API: Don't use this in client modules

def _handle_message(msg):
    """
    `msg` should be a tuple of the form: (topic, payload).

    Calls the correct handler for the topic.
    """
    if type(msg) != tuple:
        raise TypeError("Received a type {} for message. Always expecting tuple instead.".format(type(msg)))

    if msg[0] == Atom("message".encode('utf8')):
        #print("Got a strange message with 'message' as the first thing.")
        msg = msg[1]

    signal = (Atom("ok".encode('utf8')), Atom("go".encode('utf8')))
    if msg == signal:
        threading.Thread(target=_main_func).start()
    else:
        from_id, topic, msg_payload = msg  # Will throw an error here if msg is not formatted correctly
        from_id = str(from_id).lstrip('b').strip("'")
        topic = str(topic).lstrip('b').strip("'")
        _msgq.put((from_id, topic, msg_payload))

# Register the handler function with Elixir
set_message_handler(_handle_message)

