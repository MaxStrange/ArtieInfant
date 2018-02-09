"""
Module for testing the mykafka library.

This module produces a message that will be read by test1.py.
"""
import mykafka
import myargparse

def serialize(msg):
    """
    Function for serializing a msg.
    """
    return msg.encode('utf8')


if __name__ == "__main__":
    consumer_names, producer_names, consumer_configs, producer_configs = myargparse.parse_args()
    print("PRODUCING TO:", producer_names)

    mykafka.init_producer(**producer_configs)

    mykafka.produce(producer_names, key="".encode('utf8'), item="Hello!", serializer=serialize)

