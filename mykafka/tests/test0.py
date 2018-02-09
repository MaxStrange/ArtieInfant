"""
Module for testing the mykafka library.

This module will read the message that test1.py sends.
"""
import mykafka
import myargparse

def deserialize(msg):
    """
    Function for deserializing a msg.
    """
    # This is what we get back:
    # ConsumerRecord(topic='input_topic', partition=0, offset=8, timestamp=1517549968156,
    #                timestamp_type=0, key=None, value=b'wow', checksum=935428392, serialized_key_size=-1, serialized_value_size=3)
    return msg.value.decode('utf8')


if __name__ == "__main__":
    consumer_names, producer_names, consumer_configs, producer_configs = myargparse.parse_args()
    print("CONSUMING FROM:", consumer_names)

    mykafka.init_consumer(**consumer_configs)

    for msg in mykafka.consume(consumer_names, deserialize):
        print(msg)

