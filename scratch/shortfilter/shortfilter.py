"""
Filter any given wav file that is too short.
"""
import os
import pydub
import sys

MS_PER_S = 1000
S_PER_MIN = 60
MS_PER_MIN = MS_PER_S * S_PER_MIN

def shortfilter(wav_segments, duration_seconds):
    """
    Takes each wav segment and deletes it if it is less than the given length in seconds.

    Returns:
        [wav segment if wav file is longer than duration]
    """
    return [ws for ws in wav_segments if ws.duration_seconds >= duration_seconds]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3", sys.argv[0], os.sep.join(["path", "to", "file.wav"]))
        exit(-1)
    elif not os.path.isfile(sys.argv[1]):
        print(str(sys.argv[1]), "is not a valid path to a file.")
        exit(-1)

    fpath = sys.argv[1]
    filtered = shortfilter([fpath])
    print(trimmed)

