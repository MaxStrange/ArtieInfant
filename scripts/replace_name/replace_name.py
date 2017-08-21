"""
Replaces the name of each file in the given directory (recursively!) with numbers increasing
monotonically.

Only does this for wav/WAV files.
"""
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python3", sys.argv[0], os.sep.join("path to top_level directory".split(' ')))
        exit(1)

    cur = 0
    for dirname, _subs, files in os.walk(sys.argv[1]):
        for f in files:
            _name, ext = os.path.splitext(f)
            if ext.lower() == ".wav":
                newfpath = os.sep.join([dirname, str(cur) + ".WAV"])
                oldfpath = os.sep.join([dirname, f])
                cur += 1
                os.rename(oldfpath, newfpath)
