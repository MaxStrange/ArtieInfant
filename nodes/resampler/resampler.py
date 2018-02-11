"""
This node resamples the audiosegments it gets and then produces them back to whatever
topic the user requests.
"""
import audiosegment as asg
import logging
import mykafka
import myargparse

def resample(seg):
    """
    Wrapper for audiosegment_object.resample()
    def resample(self, sample_rate_Hz=None, sample_width=None, channels=None, console_output=False):
    """
    # Resample to 16bit @ 16kHz mono
    return seg.resample(sample_rate_Hz=16000, sample_width=2, channels=1)

if __name__ == "__main__":
    consumer_names, producer_names, consumer_configs, producer_configs = myargparse.parse_args()
    logging.info("CONSUMING FROM:", consumer_names, "PRODUCING TO:", producer_names)

    mykafka.init_consumer(**consumer_configs)
    mykafka.init_producer(**producer_configs)

    mykafka.consume_and_produce(consumer_names, lambda msg: asg.deserialize(msg), resample, producer_names)

