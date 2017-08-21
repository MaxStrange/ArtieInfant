"""
This script verifies that all the audio files (recursively found from the given directory) have the right format.

USAGE:
python3 <this_script.py> <path/to/top_level/directory>
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python3", sys.argv[0], os.sep.join(["path", "to", "top_level", "directory"]))
        exit(1)

    path_to_script = os.sep.join([".", "audioverifier.sh"])
    result = subprocess.run([path_to_script, sys.argv[1]], stdout=subprocess.PIPE)
    result_list = [item for item in result.stdout.decode('utf-8').split(os.linesep) if item != ""]
    channels = result_list[0::3]
    sample_rate = result_list[1::3]
    precision = result_list[2::3]
    channels_all_same = len(set(channels)) == 1
    sample_rate_all_same = len(set(sample_rate)) == 1
    precision_all_same = len(set(precision)) == 1
    chanout = "" if channels_all_same else str(set(channels))
    sampout = "" if sample_rate_all_same else str(set(sample_rate))
    precout = "" if precision_all_same else str(set(precision))
    print("Channels:", channels_all_same, chanout)
    print("Sample rates:", sample_rate_all_same, sampout)
    print("Precision:", precision_all_same, precout)
