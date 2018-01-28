"""
This library abstracts away all the work required
to plumb in the kafka library.
"""
import kafka

consumer = None
producer = None

def init_consumer(*args, **kwargs):
    """
    Must be called before consumer functions can be used.

    Merely a wrapper for calling kafka.KafkaConsumer()
    """
    consumer = kafka.KafkaConsumer(*args, **kwargs)

def init_producer(*args, **kwargs):
    """
    Must be called before producer functions can be used.

    Merely a wrapper for calling kafka.KafkaProducer()
    """
    producer = kafka.KafkaProducer(*args, **kwargs)

def consume_and_produce(consume_name, deserializer, callback, pub_name, serializer=None):
    """
    Asyncronously consumes from topic `consume_name`, spawns a thread
    to deal with the message using `callback` and then takes `callback`'s
    result and producees it asyncronously.

    :param consume_name: The name of the topic to consume.
    :param deserializer: A function that takes a kafka message as arg
                         and returns a byte string.
    :param callback:     The function to apply on the item received
                         from consuming. This must take whatever object
                         comes from calling `deserializer` on the message and
                         return an object that can be passed to `serializer`.
                         If `serializer` is None, the object must have a 'serialize'
                         method.
    :param pub_name:     The name of the topic to produce to.
    :param serializer:   Either None or a function. If None, the object returned
                         by `callback` must have a 'serialize' method that returns
                         a byte string. Otherwise, must be a function that takes
                         an object from `callback` and returns a byte string.
    """
    pass

def consume(consume_name, deserializer=None):
    """
    Syncronously consumes from topic `consume_name` and, if `deserializer` is not None,
    applies it to msgs as it gets them and yields the result. If None, the raw msg
    byte string is yielded.

    :param consume_name: The name of the topic to consume.
    :param deserializer: None or a function that takes a byte string and returns something.
    """
    pass

def produce(pub_name, item, serializer=None):
    """
    Asyncronously producees `item` to topic `pub_name`. If `serializer` is not None,
    it is applied to `item` before sending it.

    :param pub_name:    The name of the topic to produce to.
    :param item:        The item to send. If `serializer` is not None, `item` must
                        have a 'serialize' method.
    :param serializer:  Either None, in which case `item` must have a 'serialize' method,
                        or else a function that takes `item` and returns a byte string.
    """
    pass

