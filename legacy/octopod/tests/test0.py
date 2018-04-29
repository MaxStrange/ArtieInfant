"""
Module for testing the octopod library. Run with run_test_0.sh.
"""
# Import octopod from one directory up
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from octopod import octopod

# Other imports
import myargparse

if __name__ == "__main__":
    args = myargparse.parse_args()
    print("CONSUMING FROM:", args.consumer_topics)

    octopod.init_consumer(**args.consumer_configs)
    for fname, fcontents in octopod.consume_file(args.consumer_topics, args.hoststr, args.uname):
        print("Got some file contents. Writing to", fname)
        with open(fname, 'w') as f:
            f.write(fcontents)
