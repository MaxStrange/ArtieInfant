"""
Module for testing the octopod library. Run with run_test_2.sh.
"""
# Import octopod from one directory up
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from octopod import octopod

# Other imports
import myargparse
import os

if __name__ == "__main__":
    args = myargparse.parse_args()
    print("PRODUCING TO:", args.producer_topics)

    # Make a debug file
    fpath = "octopod_test_debug_file.txt"
    with open(fpath, 'w') as f:
        f.write("Blah!\nHello!")

    octopod.init_producer(**args.producer_configs)
    octopod.produce_file(args.producer_topics, key=None, thefile=fpath,
                         hoststr=args.hoststr, uname=args.uname, tmpdir=args.tmpdir)
    os.remove(fpath)
