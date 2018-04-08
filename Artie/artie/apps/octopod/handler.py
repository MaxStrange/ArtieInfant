from erlport.erlterms import Atom
from erlport.erlang import set_message_handler, cast

def register_handler(dest):
    def handler(msg):
        cast(dest, msg)
    set_message_handler(handler)
    return Atom("ok")

