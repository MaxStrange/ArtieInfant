"""
Module for testing the octopod library. Run with run_test_1.sh.
"""
# Import octopod from one directory up
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from octopod import octopod

# Other imports
import myargparse

def callback(fname, fcontents):
    """
    The callback to apply to the file contents that we from HDFS.
    Just prints the name of the file.
    """
    print("Fname:", fname)
    return fname, fcontents

if __name__ == "__main__":
    args = myargparse.parse_args()
    print("CONSUMING FROM:", args.consumer_topics, "PRODUCING TO:", args.producer_topics)

    octopod.init_consumer(**args.consumer_configs)
    octopod.init_producer(**args.producer_configs)

    # Runs forever - accepts files, filters the silence in them, then publishes them
    octopod.consume_and_produce(args.consumer_topics, callback, args.producer_topics,
                                args.hoststr, args.uname, args.tmpdir)
