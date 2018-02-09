"""
Module for testing the mykafka library.
"""
import mykafka
import myargparse

def do_a_thing_with_a_msg(msg):
    """
    This is the function that will get called when the msg gets received and deserialized.
    It must return a thing that can be serialized.
    """
    print(msg)
    return msg

def deserialize(msg):
    """
    Function for deserializing a msg.
    """
    #ConsumerRecord(topic='input_topic', partition=0, offset=8, timestamp=1517549968156, timestamp_type=0, key=None, value=b'wow', checksum=935428392, serialized_key_size=-1, serialized_value_size=3)
    return msg.value.decode('utf8')

def serialize(msg):
    """
    Function for serializing a msg.
    """
    return msg.encode('utf8')


if __name__ == "__main__":
    consumer_names, producer_names, consumer_configs, producer_configs = myargparse.parse_args()
    print("CONSUMING FROM:", consumer_names, "PRODUCING TO:", producer_names)

    mykafka.init_consumer(**consumer_configs)
    mykafka.init_producer(**producer_configs)

    # Runs forever - accepts messages from consumer_names, filters the silence, then publishes to producer_names
    mykafka.consume_and_produce(consumer_names, deserialize, do_a_thing_with_a_msg, producer_names, serializer=serialize)

