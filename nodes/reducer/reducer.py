"""
This node waits for audio segments and reduces any whose names
are the same into a single audiosegment before saving that one
to disk.
"""
import audiosegment as asg
import logging
import mykafka
import myargparse

def reduce_and_save(consumer_names):
    cached = {}
    for seg in mykafka.consume(consumer_names, deserializer=lambda msg: asg.deserialize(msg.value)):
        if seg.name in cached:
            cached[seg.name] = cached[seg.name].reduce(seg)
        else:
            cached[seg.name] = seg

if __name__ == "__main__":
    consumer_names, producer_names, consumer_configs, producer_configs = myargparse.parse_args()
    logging.info("CONSUMING FROM:", consumer_names)

    mykafka.init_consumer(**consumer_configs)

    reduce_and_save(consumer_names)

