"""
This is a Kafka node that subscribes to whatever topics are passed in
and publishes to whatever topics are passed in. Call 'help' to see
options.
"""
import audiosegment as asg
import logging
import mykafka
import myargparse

def remove_silence(seg):
    """
    Wrapper for audiosegment_object.filter_silence()
    """
    return seg.filter_silence()

if __name__ == "__main__":
    consumer_names, producer_names, consumer_configs, producer_configs = myargparse.parse_args()
    logging.info("CONSUMING FROM:", consumer_names, "PRODUCING TO:", producer_names)

    mykafka.init_consumer(**consumer_configs)
    mykafka.init_producer(**producer_configs)

    # Runs forever - accepts messages from consumer_names, filters the silence, then publishes to producer_names
    mykafka.consume_and_produce(consumer_names, lambda msg: asg.deserialize(msg.value), remove_silence, producer_names)

