"""
This library abstracts away all the work required
to plumb in the kafka library.
"""
import atexit
import kafka
import queue
import threading

mailbox = queue.Queue()
consumer = None
producer = None

class NotInitializedException(Exception):
    pass

def _at_close():
    """
    Cleanup function.
    """
    if consumer is not None:
        print("Closing consumer...")
        consumer.close()
    if producer is not None:
        print("Closing producer...")
        producer.close()

def _repair_kwargs(kw):
    """
    Takes configurations and changes their names to whatever the kafka library is actually expecting.
    """
    # - bootstrap-server is 'bootstrap-servers' in this library, so change it if you see it
    # - 'api-version' needs to be added or things don't work with remote brokers
    newkw = {}
    for k, v in kw.items():
        if k == "bootstrap_server":
            newkw["bootstrap_servers"] = v
        else:
            newkw[k] = v
    if 'api_version' not in newkw:
        newkw['api_version'] = (0, 10)

    return newkw

def init_consumer(*args, **kwargs):
    """
    Must be called before consumer functions can be used.

    Merely a wrapper for calling kafka.KafkaConsumer()
    """
    global consumer
    kwargs = _repair_kwargs(kwargs)
    consumer = kafka.KafkaConsumer(*args, **kwargs)
    atexit.register(_at_close)

def init_producer(*args, **kwargs):
    """
    Must be called before producer functions can be used.

    Merely a wrapper for calling kafka.KafkaProducer()
    """
    global producer
    kwargs = _repair_kwargs(kwargs)
    producer = kafka.KafkaProducer(*args, **kwargs)
    atexit.register(_at_close)

def _callback_and_enqueue(callback, msg, deserializer):
    """
    Executes `callback` on `msg` and stores the result in mailbox.
    """
    deserialized = deserializer(msg)
    ret = callback(deserialized)
    mailbox.put(ret)

def _wait_for_msgs(callback, deserializer):
    """
    Polls the consumer and upon getting a msg, spawns a thread with `callback`.
    """
    assert consumer is not None
    for msg in consumer:
        t = threading.Thread(target=_callback_and_enqueue, args=(callback, msg, deserializer))
        t.start()

def _produce_from_mailbox(pub_names, serializer):
    """
    Blocks until mailbox has something, then calls `serializer` on it (if
    `serializer` is not None (if it is, attempts to call the object's
    'serialize()' method), then calls producer.send() on the result.
    """
    while True:
        item = mailbox.get()
        serialized = serializer(item) if serializer is not None else item.serialize()
        for name in pub_names:
            try:
                key, value = serialized
                producer.send(name, value, key)
            except (TypeError, ValueError):
                producer.send(name, serialized)

def consume_and_produce(consume_names, deserializer, callback, pub_names, serializer=None):
    """
    Asyncronously consumes from topic `consume_name`, spawns a thread
    to deal with the message using `callback` and then takes `callback`'s
    result and producees it asyncronously.

    :param consume_names:   The names of the topics to consume. Must be iterable.
    :param deserializer:    A function that takes a kafka message as arg
                            and returns a byte string.
    :param callback:        The function to apply on the item received
                            from consuming. This must take whatever object
                            comes from calling `deserializer` on the message and
                            return an object that can be passed to `serializer`.
                            If `serializer` is None, the object must have a 'serialize'
                            method.
    :param pub_names:       The names of the topics to produce to. Must be iterable.
    :param serializer:      Either None or a function. If None, the object returned
                            by `callback` must have a 'serialize' method that returns
                            a byte string. Otherwise, must be a function that takes
                            an object from `callback` and returns a byte string.
                            Serialized may return a key, value pair or a single item.
                            If it returns a single item, it is treated as the value, and
                            the partition will be chosen randomly.
    """
    if consumer is None:
        raise NotInitializedException("You must call init_consumer() first.")
    if producer is None:
        raise NotInitializedException("You must call init_producer() first.")


    consumer.subscribe(consume_names)
    consumer_thread = threading.Thread(target=_wait_for_msgs, args=(callback, deserializer))
    consumer_thread.start()

    producer_thread = threading.Thread(target=_produce_from_mailbox, args=(pub_names, serializer))
    producer_thread.start()

def consume(consume_names, deserializer=None):
    """
    Syncronously consumes from topic `consume_name` and, if `deserializer` is not None,
    applies it to msgs as it gets them and yields the result. If None, the raw msg
    byte string is yielded.

    :param consume_names: The names of the topics to consume. Must be iterable.
    :param deserializer:  None or a function that takes a byte string and returns something.
    """
    consumer.subscribe(consume_names)
    for msg in consumer:
        if deserializer is not None:
            yield deserializer(msg)
        else:
            yield msg

def produce(pub_names, key, item, serializer=None):
    """
    Syncronously producees `item` to each topic in `pub_names`. If `serializer` is not None,
    it is applied to `item` before sending it.

    :param pub_names:   The names of the topics to produce to. Must be iterable.
    :param key:         The key in the key, value pair. Can be None, in which case,
                        a partition is chosen at random to write to.
    :param item:        The item to send. If `serializer` is not None, `item` must
                        have a 'serialize' method.
    :param serializer:  Either None, in which case `item` must have a 'serialize' method,
                        or else a function that takes `item` and returns a byte string.
    """
    for name in pub_names:
        v = serializer(item) if serializer is not None else item.serialize()
        producer.send(name, value=v, key=key)

