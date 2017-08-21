"""
USAGE:
python3 <this_script.py> <path/to/top_level/dir>

This script converts ALL wav/WAV files RECURSIVELY to:
- 16 bit
- 24 kHz
- Mono
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python3", sys.argv[0], os.sep.join(["path", "to", "top_level", "directory"]))
        exit(1)

    for dirname, subdirs, files in os.walk(sys.argv[1]):
        for f in files:
            fpath = os.sep.join([dirname, f])
            fname, ext = os.path.splitext(fpath)
            if ext.lower() == ".wav":
                print("Converting", fpath, "to 16 bit, one channel audio sampled at 24kHz.")
                outname = fname + "output.WAV"
                command = "sox " + fpath + " -b16 -r 24000 " + outname + " channels 1"
                res = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
                res.check_returncode()
                os.rename(outname, fpath)
