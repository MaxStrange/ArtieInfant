"""
This node waits for audio segments and reduces any whose names
are the same into a single audiosegment before saving that one
to disk.
"""
import audiosegment as asg
import logging
import mykafka
import myargparse
import os

def reduce_and_save(consumer_names):
    cached = {}
    for seg in mykafka.consume(consumer_names, deserializer=lambda msg: asg.deserialize(msg.value)):
        logging.info("Got seg.")
        if seg.name in cached:
            cached[seg.name] = cached[seg.name].reduce([seg])
        else:
            cached[seg.name] = seg
        # Save the file so far created - we will overwrite it later if we encounter the same name again
        basename_no_ext, _ext = os.path.splitext(os.path.basename(seg.name))
        save_name = "results" + os.sep + basename_no_ext + ".wav"
        logging.info("Saving: " + save_name)
        cached[seg.name].export(save_name, format="WAV")

if __name__ == "__main__":
    consumer_names, producer_names, consumer_configs, producer_configs = myargparse.parse_args()
    logging.basicConfig(level="INFO")
    logging.info("CONSUMING FROM: " + str(consumer_names))

    mykafka.init_consumer(**consumer_configs)

    reduce_and_save(consumer_names)

